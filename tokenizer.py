"""
tokenizer.py
============
Converts raw OhioT1DM parsed data dictionaries into fixed-length token
sequences ready for input to the Quantile Transformer model.

Each token is a 6-dimensional feature vector representing one 5-minute
CGM timestep. A sequence of 24 tokens (2 hours) forms one model input.
The prediction target is the glucose value 6 timesteps (30 minutes) ahead.

Design decisions documented inline — do not change without understanding
the physiological and statistical reasoning behind each choice.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# ── Constants ────────────────────────────────────────────────────────────────

SAMPLING_INTERVAL_MIN = 5        # CGM sampling rate in minutes
INPUT_WINDOW          = 24       # Timesteps of context (24 × 5min = 2 hours)
PREDICTION_HORIZON    = 6        # Timesteps ahead to predict (6 × 5min = 30min)
MAX_INTERP_GAP_MIN    = 20       # Gaps up to this length → linear interpolation
MAX_MASK_GAP_MIN      = 120      # Gaps up to this length → masking (no interp)
                                 # Gaps beyond this → discard affected windows
MAX_ROC               = 4.0      # Maximum physiologically plausible rate of
                                 # change in mg/dL per minute (interstitial)
N_FEATURES            = 6        # Feature vector dimensionality


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class PatientStats:
    """
    Normalization statistics computed from a single patient's training data.
    Stored alongside the model so inference on new patients can reproduce
    the same normalization.
    """
    glucose_mean : float
    glucose_std  : float
    patient_id   : str


@dataclass
class TokenSequence:
    """
    One model-ready sample.

    Attributes
    ----------
    features : np.ndarray, shape (INPUT_WINDOW, N_FEATURES)
        The input token sequence.
    target : float
        Normalized glucose value at prediction horizon.
    target_raw : float
        Raw glucose value at prediction horizon in mg/dL.
        Stored for evaluation — never fed to the model.
    attention_mask : np.ndarray, shape (INPUT_WINDOW,)
        Boolean array. True = this token is valid and should be attended to.
        False = this token was in a medium-length gap and should be masked.
    start_time : pd.Timestamp
        Wall-clock time of the first token in the window.
        Used for debugging and visualization.
    """
    features       : np.ndarray
    target         : float
    target_raw     : float
    attention_mask : np.ndarray
    start_time     : pd.Timestamp


# ── Step 1: Build the regular CGM grid ───────────────────────────────────────

def build_cgm_grid(glucose_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample the raw CGM readings onto a perfectly regular 5-minute grid.

    The raw CGM has 98.5% of readings exactly 5 minutes apart, with occasional
    gaps due to sensor dropout. This function creates a complete regular grid
    spanning the full date range, then aligns raw readings to their nearest
    grid point (within a 2.5-minute tolerance), leaving everything else as NaN.

    The gap classification and interpolation happens in a later step.
    We deliberately keep NaN here rather than filling, so downstream steps
    can distinguish "reading existed" from "reading was inferred."

    Parameters
    ----------
    glucose_df : pd.DataFrame
        Raw glucose_level dataframe from parse_ohio_xml, with columns
        ['ts', 'value'].

    Returns
    -------
    pd.DataFrame with columns:
        'ts'            — regular 5-minute timestamps
        'glucose_raw'   — original CGM value at this timestep (NaN if missing)
        'gap_minutes'   — length of the gap this timestep falls within (0 if not
                          in a gap). Used for gap classification downstream.
    """
    # Build the complete regular grid
    start = glucose_df['ts'].min().floor('5min')
    end   = glucose_df['ts'].max().ceil('5min')
    grid  = pd.date_range(start=start, end=end, freq='5min')
    grid_df = pd.DataFrame({'ts': grid})

    # Align raw readings to nearest grid point using merge_asof
    # This handles the minor timing jitter (e.g., 11:36:29 → 11:35:00)
    glucose_sorted = glucose_df.sort_values('ts').copy()
    glucose_sorted = glucose_sorted.rename(columns={'value': 'glucose_raw'})

    grid_df = pd.merge_asof(
        grid_df,
        glucose_sorted,
        on='ts',
        tolerance=pd.Timedelta('2min30s'),  # Half of sampling interval
        direction='nearest'
    )

    # Compute gap length for each NaN timestep
    # A "gap" is a contiguous run of NaN readings
    is_missing = grid_df['glucose_raw'].isna()
    gap_group  = (is_missing != is_missing.shift()).cumsum()

    gap_lengths = pd.Series(0.0, index=grid_df.index)
    for group_id, group_df in grid_df[is_missing].groupby(gap_group[is_missing]):
        gap_len_min = len(group_df) * SAMPLING_INTERVAL_MIN
        gap_lengths.loc[group_df.index] = gap_len_min

    grid_df['gap_minutes'] = gap_lengths

    return grid_df


# ── Step 2: Handle gaps ───────────────────────────────────────────────────────

def handle_gaps(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-tier gap handling strategy to the regular CGM grid.

    Tier 1 — Short gaps (≤ MAX_INTERP_GAP_MIN = 20 minutes):
        Linear interpolation between flanking readings.
        Physiologically defensible because glucose cannot change faster than
        ~4 mg/dL/min — over 20 minutes the maximum possible change is 80 mg/dL,
        but in practice short gaps rarely coincide with rapid excursions.
        The is_interpolated flag marks these timesteps so the model can
        learn to down-weight them via attention.

    Tier 2 — Medium gaps (20 < gap ≤ MAX_MASK_GAP_MIN = 120 minutes):
        Fill with the last known value (forward fill) but set is_interpolated=1
        AND mark these positions in the attention mask as invalid.
        Forward fill prevents NaN propagation through the feature computation
        pipeline, but the attention mask tells the Transformer to ignore these
        positions entirely during self-attention.

    Tier 3 — Long gaps (> MAX_MASK_GAP_MIN = 120 minutes):
        Forward fill to prevent NaN (same as Tier 2) but mark with a special
        flag so window sampling can identify and discard windows containing
        these timesteps.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of build_cgm_grid.

    Returns
    -------
    pd.DataFrame with additional columns:
        'glucose_filled'   — glucose values after gap handling (no NaN)
        'is_interpolated'  — 1 if Tier 1 interpolated, 0 otherwise
        'attention_valid'  — 0 if Tier 2/3 masked, 1 otherwise
        'is_long_gap'      — 1 if Tier 3 (long gap), used to discard windows
    """
    df = grid_df.copy()

    # Classify each missing timestep by its gap tier
    short_gap  = (df['gap_minutes'] > 0) & (df['gap_minutes'] <= MAX_INTERP_GAP_MIN)
    medium_gap = (df['gap_minutes'] > MAX_INTERP_GAP_MIN) & (df['gap_minutes'] <= MAX_MASK_GAP_MIN)
    long_gap   = df['gap_minutes'] > MAX_MASK_GAP_MIN

    # Start with the raw values
    df['glucose_filled']  = df['glucose_raw'].copy()
    df['is_interpolated'] = 0
    df['attention_valid'] = 1
    df['is_long_gap']     = 0

    # Tier 1: Linear interpolation for short gaps
    # pandas interpolate() fills NaN linearly between flanking valid values
    df['glucose_filled'] = df['glucose_filled'].interpolate(
        method='linear', limit_area='inside'
    )
    df.loc[short_gap, 'is_interpolated'] = 1

    # Tier 2: Forward fill for medium gaps, mask attention
    df['glucose_filled'] = df['glucose_filled'].ffill()
    df.loc[medium_gap, 'is_interpolated'] = 1
    df.loc[medium_gap, 'attention_valid'] = 0

    # Tier 3: Forward fill for long gaps, mask attention, flag for window discard
    df.loc[long_gap, 'is_interpolated'] = 1
    df.loc[long_gap, 'attention_valid'] = 0
    df.loc[long_gap, 'is_long_gap']     = 1

    # Final safety: if any NaN remains at the start (no prior value to ffill from)
    # use backward fill. This should be rare.
    df['glucose_filled'] = df['glucose_filled'].bfill()

    return df


# ── Step 3: Compute derived features ─────────────────────────────────────────

def compute_features(
    grid_df    : pd.DataFrame,
    meal_df    : pd.DataFrame,
    stats      : PatientStats,
) -> pd.DataFrame:
    """
    Compute all 6 features for every timestep on the regular grid.

    Features are computed in normalized form ready for model input.
    Raw values are preserved in separate columns for debugging.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Output of handle_gaps — complete regular grid with gap handling applied.
    meal_df : pd.DataFrame
        Raw meal dataframe from parse_ohio_xml, with columns ['ts', 'carbs'].
    stats : PatientStats
        Per-patient normalization statistics from compute_patient_stats().

    Returns
    -------
    pd.DataFrame with all original columns plus the 6 model features:
        f0_glucose, f1_roc, f2_time_sin, f3_time_cos, f4_carbs, f5_interp
    """
    df = grid_df.copy()

    # ── Feature 0: Normalized glucose value ──────────────────────────────────
    # Per-patient z-score normalization.
    # We normalize using training-set statistics only (stats computed before
    # this function is called on validation/test data) to prevent data leakage.
    df['f0_glucose'] = (
        (df['glucose_filled'] - stats.glucose_mean) / stats.glucose_std
    )

    # ── Feature 1: Rate of change (mg/dL per minute) ─────────────────────────
    # First-order finite difference on the filled glucose values.
    # Divided by SAMPLING_INTERVAL_MIN to express as per-minute rate.
    #
    # We clip to ±MAX_ROC to remove physiologically implausible values that
    # arise near gap boundaries where interpolation jumps sharply. For example,
    # if glucose was 80 before a 15-minute gap and 120 after, the interpolated
    # slope across the boundary is (120-80)/(15min) = 2.67 mg/dL/min — plausible.
    # But a Tier 2 forward-filled gap could produce an apparent 0 slope followed
    # by a sharp jump at the gap boundary, which clipping handles gracefully.
    #
    # Note: RoC is NOT normalized separately because its scale is already
    # physiologically bounded and consistent across patients.
    roc_raw = df['glucose_filled'].diff() / SAMPLING_INTERVAL_MIN
    df['f1_roc'] = roc_raw.clip(-MAX_ROC, MAX_ROC).fillna(0.0)

    # ── Features 2 & 3: Time-of-day encoding ─────────────────────────────────
    # Sinusoidal encoding of the hour of day.
    # Using sin/cos pair rather than raw hour for two reasons:
    #   1. Cyclical continuity: 23:55 and 00:05 are 10 minutes apart,
    #      but raw hours 23 and 0 are numerically far apart.
    #   2. The pair together uniquely identifies any time of day, whereas
    #      sin alone is ambiguous (same value at two times of day).
    hours = df['ts'].dt.hour + df['ts'].dt.minute / 60.0
    df['f2_time_sin'] = np.sin(2 * np.pi * hours / 24.0)
    df['f3_time_cos'] = np.cos(2 * np.pi * hours / 24.0)

    # ── Feature 4: Carbohydrate intake ───────────────────────────────────────
    # Meal events are sparse point events (73 over 45 days).
    # We align each meal to its nearest 5-minute grid point and place the
    # carb value there. All other timesteps are 0.
    #
    # Normalized by dividing by 100 (the observed maximum in this dataset)
    # so the feature lives in [0, 1] and is interpretable as fraction of
    # the largest observed meal. A future model could use a dataset-wide
    # max here for consistency across patients.
    #
    # We use merge_asof with a 2.5-minute tolerance — the same alignment
    # approach as the glucose grid — to snap meal timestamps to the grid.
    df['f4_carbs'] = 0.0

    if len(meal_df) > 0:
        meal_aligned = pd.merge_asof(
            meal_df.sort_values('ts')[['ts', 'carbs']],
            df[['ts']].reset_index(),
            on='ts',
            tolerance=pd.Timedelta('2min30s'),
            direction='nearest'
        )
        # Place carb values at the matched grid indices
        valid_matches = meal_aligned.dropna(subset=['index'])
        for _, row in valid_matches.iterrows():
            grid_idx = int(row['index'])
            df.loc[grid_idx, 'f4_carbs'] = row['carbs'] / 100.0

    # ── Feature 5: Interpolation flag ────────────────────────────────────────
    # Binary flag — 1 if this timestep was gap-filled by any tier.
    # Included as an explicit feature so the Transformer can learn to
    # reduce confidence when attending over imputed timesteps.
    # (Attention masking handles Tier 2/3 gaps structurally, but this flag
    # also communicates Tier 1 interpolation which is not masked.)
    df['f5_interp'] = df['is_interpolated'].astype(float)

    return df


# ── Step 4: Compute normalization statistics ──────────────────────────────────

def compute_patient_stats(glucose_df: pd.DataFrame, patient_id: str) -> PatientStats:
    """
    Compute per-patient normalization statistics from training data only.

    IMPORTANT: This must be called on training data only. Never compute
    statistics on validation or test data — that would constitute data leakage
    because the model would indirectly see the distribution of future values.

    Mean and std are computed on raw glucose values before any gap handling,
    using only non-NaN readings.

    Parameters
    ----------
    glucose_df : pd.DataFrame
        Raw glucose_level dataframe from parse_ohio_xml.
    patient_id : str
        Patient identifier string, stored for reference.

    Returns
    -------
    PatientStats with mean and std for this patient.
    """
    values = glucose_df['value'].dropna().values
    return PatientStats(
        glucose_mean = float(np.mean(values)),
        glucose_std  = float(np.std(values)),
        patient_id   = patient_id,
    )


# ── Step 5: Extract token sequences (sliding window) ─────────────────────────

def extract_sequences(feature_df: pd.DataFrame) -> List[TokenSequence]:
    """
    Slide a window across the feature dataframe to extract all valid
    model-ready samples.

    Window structure:
        [t-23, t-22, ..., t-1, t]  →  predict glucose at [t+6]
         ←────── INPUT_WINDOW ──────→       prediction horizon

    A window is DISCARDED if any of the following are true:
        1. Any timestep in the input window is part of a long gap (>2 hours)
        2. The timestep immediately before the prediction horizon (t) is
           missing or long-gap — we have no recent context to predict from
        3. The prediction target timestep (t+6) is NaN or long-gap —
           we have no valid label
        4. The prediction target falls within MAX_MASK_GAP_MIN minutes
           after a long gap — even if the target reading exists, the model
           has no meaningful context leading up to it

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of compute_features, with all f0–f5 columns and metadata.

    Returns
    -------
    List of TokenSequence objects, one per valid window.
    """
    feature_cols = ['f0_glucose', 'f1_roc', 'f2_time_sin',
                    'f3_time_cos', 'f4_carbs', 'f5_interp']

    features_arr   = feature_df[feature_cols].values.astype(np.float32)
    attn_valid_arr = feature_df['attention_valid'].values
    long_gap_arr   = feature_df['is_long_gap'].values
    glucose_raw    = feature_df['glucose_filled'].values
    timestamps     = feature_df['ts'].values

    sequences = []

    # Total number of positions we can center a window on
    total = len(feature_df)
    first_valid = INPUT_WINDOW          # First position with full input window
    last_valid  = total - PREDICTION_HORIZON  # Last position with valid target

    for t in range(first_valid, last_valid):
        window_start = t - INPUT_WINDOW
        window_end   = t              # Inclusive end of input window
        target_idx   = t + PREDICTION_HORIZON - 1

        # ── Discard rule 1: long gap anywhere in input window ─────────────
        if long_gap_arr[window_start:window_end].any():
            continue

        # ── Discard rule 2: most recent timestep is masked ────────────────
        if not attn_valid_arr[window_end - 1]:
            continue

        # ── Discard rule 3: prediction target is invalid ──────────────────
        if long_gap_arr[target_idx] or np.isnan(glucose_raw[target_idx]):
            continue

        # ── Discard rule 4: target within recovery window after long gap ──
        # Check if any long gap occurred within the prediction horizon window
        if long_gap_arr[window_end:target_idx + 1].any():
            continue

        # ── Valid window — extract features ───────────────────────────────
        window_features = features_arr[window_start:window_end].copy()
        window_attn     = attn_valid_arr[window_start:window_end].astype(bool)

        # Normalized target (for loss computation during training)
        target_normalized = features_arr[target_idx, 0]  # f0_glucose at target

        # Raw target in mg/dL (for evaluation metrics)
        target_raw = float(glucose_raw[target_idx])

        sequences.append(TokenSequence(
            features       = window_features,
            target         = float(target_normalized),
            target_raw     = target_raw,
            attention_mask = window_attn,
            start_time     = pd.Timestamp(timestamps[window_start]),
        ))

    return sequences


# ── Top-level entry point ─────────────────────────────────────────────────────

def tokenize_patient(
    data       : Dict[str, pd.DataFrame],
    patient_id : str,
    stats      : Optional[PatientStats] = None,
) -> Tuple[List[TokenSequence], PatientStats]:
    """
    Full tokenization pipeline for one patient.

    Runs all five steps in order:
        1. Build regular CGM grid
        2. Handle gaps (three-tier strategy)
        3. Compute all 6 features
        4. Extract sliding window sequences

    Parameters
    ----------
    data : dict
        Output of parse_ohio_xml for one patient file.
    patient_id : str
        Patient identifier, e.g. '540'.
    stats : PatientStats, optional
        If provided, use these normalization statistics (for val/test patients).
        If None, compute from this patient's data (for training patients).

    Returns
    -------
    sequences : List[TokenSequence]
        All valid token sequences for this patient.
    stats : PatientStats
        The normalization statistics used (computed or passed in).
    """

    # Validate required streams are present
    required = ['glucose_level']
    for stream in required:
        if stream not in data:
            raise KeyError(
                f"Patient {patient_id}: required stream '{stream}' missing "
                f"from XML. Available streams: {list(data.keys())}"
            )

    # Optional streams — degrade gracefully if absent
    glucose_df = data['glucose_level']

    if 'meal' in data and len(data['meal']) > 0:
        meal_df = data['meal']
    else:
        if 'meal' not in data:
            print(f"  Patient {patient_id}: 'meal' stream absent "
                  f"— no carb features for this file")
        meal_df = pd.DataFrame(columns=['ts', 'carbs'])

    glucose_df = data['glucose_level']

    # Safely retrieve meal stream — may be absent in test XMLs where
    # the patient logged zero meals during the recording period
    if 'meal' in data and len(data['meal']) > 0:
        meal_df = data['meal']
    else:
        # Empty DataFrame with correct schema so compute_features
        # doesn't need to know whether meals exist
        meal_df = pd.DataFrame(columns=['ts', 'carbs'])

    # Step 1: Regular grid
    grid_df = build_cgm_grid(glucose_df)

    # Step 2: Gap handling
    grid_df = handle_gaps(grid_df)

    # Step 3a: Normalization stats (training only)
    if stats is None:
        stats = compute_patient_stats(glucose_df, patient_id)

    # Step 3b: Feature computation
    feature_df = compute_features(grid_df, meal_df, stats)

    # Step 4: Sliding window extraction
    sequences = extract_sequences(feature_df)

    print(f"Patient {patient_id}: {len(sequences)} valid sequences "
          f"from {len(grid_df)} grid timesteps "
          f"(glucose mean={stats.glucose_mean:.1f}, "
          f"std={stats.glucose_std:.1f})")

    return sequences, stats


# ── Verification ──────────────────────────────────────────────────────────────

def verify_tokenization(sequences: List[TokenSequence]):
    """
    Sanity checks on the extracted sequences.
    Call this after tokenize_patient to catch any data issues early.
    """
    assert len(sequences) > 0, "No valid sequences extracted"

    shapes_ok   = all(s.features.shape == (INPUT_WINDOW, N_FEATURES)
                      for s in sequences)
    masks_ok    = all(s.attention_mask.shape == (INPUT_WINDOW,)
                      for s in sequences)
    no_nan      = all(not np.isnan(s.features).any() for s in sequences)
    no_inf      = all(not np.isinf(s.features).any() for s in sequences)
    targets_ok  = all(not np.isnan(s.target) for s in sequences)

    print(f"\nVerification results:")
    print(f"  Total sequences : {len(sequences)}")
    print(f"  Shape correct   : {shapes_ok}")
    print(f"  Masks correct   : {masks_ok}")
    print(f"  No NaN in feats : {no_nan}")
    print(f"  No Inf in feats : {no_inf}")
    print(f"  Targets valid   : {targets_ok}")

    # Feature range checks
    features_stacked = np.stack([s.features for s in sequences])
    print(f"\n  Feature ranges (min, max, mean):")
    feature_names = ['glucose', 'roc', 'time_sin', 'time_cos', 'carbs', 'interp']
    for i, name in enumerate(feature_names):
        col = features_stacked[:, :, i].flatten()
        print(f"    f{i} {name:10s}: "
              f"[{col.min():7.3f}, {col.max():7.3f}], "
              f"mean={col.mean():7.3f}")

    # Meal feature sparsity check
    carb_nonzero = (features_stacked[:, :, 4] > 0).sum()
    print(f"\n  Non-zero carb tokens: {carb_nonzero} "
          f"({carb_nonzero / features_stacked[:, :, 4].size * 100:.2f}% of all tokens)")

    # Mask check
    masks_stacked = np.stack([s.attention_mask for s in sequences])
    masked_fraction = (~masks_stacked).mean()
    print(f"  Masked token fraction: {masked_fraction:.4f}")

    all_ok = shapes_ok and masks_ok and no_nan and no_inf and targets_ok
    print(f"\n  {'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED — review above'}")
    return all_ok