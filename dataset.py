"""
dataset.py
==========
PyTorch Dataset and DataLoader construction for the OhioT1DM population model.

Responsibilities:
    - Load and tokenize all patients from train/test XML directories
    - Split training data chronologically into train and validation sets
    - Expose a unified population Dataset that mixes patients
    - Provide a per-patient mode for future per-patient evaluation
    - Return attention masks alongside features so the Transformer
      can ignore gap-masked tokens during self-attention

Split structure:
    Training XML  → first TRAIN_FRAC  → train sequences
                  → remaining         → validation sequences
    Testing XML   → 100%              → test sequences (one per patient)

The boundary buffer discards sequences whose prediction target crosses
the train/validation boundary to prevent any label leakage.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from tokenizer import (
    tokenize_patient,
    verify_tokenization,
    TokenSequence,
    PatientStats,
    INPUT_WINDOW,
    N_FEATURES,
    PREDICTION_HORIZON,
)


# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_FRAC = 0.70   # Fraction of each patient's training XML used for training
                    # Remainder goes to validation
                    # Must be chronological — never shuffle across this boundary


# ── XML parser (reproduced here for self-containment) ────────────────────────

def parse_ohio_xml(filepath: str) -> Dict[str, pd.DataFrame]:
    """Parse one OhioT1DM XML file into a dict of DataFrames."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(filepath)
    root = tree.getroot()

    dfs = {}
    for sensor in root:
        records = [event.attrib for event in sensor.findall("event")]
        if records:
            df = pd.DataFrame(records)
            # Normalise timestamp column names across streams
            for ts_col in ['ts', 'ts_begin', 'tbegin']:
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(
                        df[ts_col], format='%d-%m-%Y %H:%M:%S'
                    )
            for ts_col in ['ts_end', 'tend']:
                if ts_col in df.columns:
                    df[ts_col] = pd.to_datetime(
                        df[ts_col], format='%d-%m-%Y %H:%M:%S'
                    )
            for col in df.columns:
                if col not in ['ts', 'ts_begin', 'ts_end',
                               'tbegin', 'tend', 'type']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            dfs[sensor.tag] = df

    return dfs


# ── Per-patient split helper ──────────────────────────────────────────────────

def split_sequences_chronological(
    sequences : List[TokenSequence],
    train_frac: float = TRAIN_FRAC,
) -> Tuple[List[TokenSequence], List[TokenSequence]]:
    """
    Split a chronologically ordered list of TokenSequences into train and
    validation sets, with a boundary buffer to prevent target leakage.

    The sequences are already in chronological order from extract_sequences()
    because we slide the window forward through time. We do NOT shuffle before
    splitting — that would constitute temporal data leakage.

    Boundary buffer:
        Sequences near the split boundary are discarded if their prediction
        target (INPUT_WINDOW + PREDICTION_HORIZON - 1 timesteps from the
        sequence start) falls within the validation period. In practice this
        removes PREDICTION_HORIZON = 6 sequences from the end of training.

    Parameters
    ----------
    sequences : list of TokenSequence
        Chronologically ordered sequences for one patient.
    train_frac : float
        Fraction of sequences assigned to training.

    Returns
    -------
    train_seqs, val_seqs : lists of TokenSequence
    """
    n = len(sequences)
    split_idx = int(n * train_frac)

    # Apply boundary buffer — discard the PREDICTION_HORIZON sequences
    # immediately before the split boundary from the training side
    # These sequences have their target timestep in the validation period
    buffer = PREDICTION_HORIZON
    train_seqs = sequences[:max(0, split_idx - buffer)]
    val_seqs   = sequences[split_idx:]

    return train_seqs, val_seqs


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class GlucoseDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of TokenSequence objects.

    Each item returned is a dict with keys:
        'features'       : FloatTensor (INPUT_WINDOW, N_FEATURES)
        'attention_mask' : BoolTensor  (INPUT_WINDOW,)
                           True  = valid token, attend normally
                           False = gap-masked token, ignore in attention
        'target'         : FloatTensor (1,)
                           Normalized glucose value at prediction horizon
        'target_raw'     : FloatTensor (1,)
                           Raw glucose in mg/dL — for evaluation only,
                           never fed to the model
        'patient_id'     : str
                           Patient identifier — enables per-patient
                           filtering and evaluation downstream

    Design note — why a dict rather than a tuple:
        Returning named fields makes the training loop explicit about what
        it's consuming. When we add bolus features later, the dict schema
        extends cleanly without breaking existing training code.

    Design note — per-patient extensibility:
        patient_id is stored on every sample. The DataLoader collates
        these into a list of strings per batch. Downstream code can use
        this to compute per-patient metrics without needing a separate
        data structure.
    """

    def __init__(
        self,
        sequences  : List[TokenSequence],
        patient_id : str,
        stats      : PatientStats,
    ):
        self.sequences  = sequences
        self.patient_id = patient_id
        self.stats      = stats   # Stored for inverse normalization at eval time

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]

        return {
            'features'      : torch.tensor(seq.features,
                                            dtype=torch.float32),
            'attention_mask': torch.tensor(seq.attention_mask,
                                            dtype=torch.bool),
            'target'        : torch.tensor([seq.target],
                                            dtype=torch.float32),
            'target_raw'    : torch.tensor([seq.target_raw],
                                            dtype=torch.float32),
            'patient_id'    : self.patient_id,
        }


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function that handles the mixed-type batch dict.

    PyTorch's default collate handles tensors fine but stumbles on the
    patient_id strings. This collate stacks tensors normally and collects
    patient_ids as a plain list.

    The attention_mask convention follows HuggingFace Transformers:
        True  = token is valid (attend to it)
        False = token is masked (ignore it)

    Note: PyTorch's nn.TransformerEncoder uses the opposite convention for
    src_key_padding_mask — True means IGNORE. We flip the mask when passing
    to the Transformer in the model code, not here. Keeping the intuitive
    convention (True = valid) in the Dataset makes downstream code easier
    to reason about.
    """
    return {
        'features'      : torch.stack([b['features']       for b in batch]),
        'attention_mask': torch.stack([b['attention_mask']  for b in batch]),
        'target'        : torch.stack([b['target']          for b in batch]),
        'target_raw'    : torch.stack([b['target_raw']      for b in batch]),
        'patient_id'    : [b['patient_id'] for b in batch],
    }


# ── Top-level data loading ────────────────────────────────────────────────────

@dataclass
class DataSplit:
    """
    Container for all DataLoaders and per-patient metadata.

    Attributes
    ----------
    train_loader : DataLoader
        Population training loader — all patients interleaved, shuffled.
    val_loader : DataLoader
        Population validation loader — all patients, no shuffle.
    test_loaders : dict of str → DataLoader
        One DataLoader per patient for the test set.
        Keyed by patient_id string, e.g. '540'.
        Evaluated separately to report per-patient metrics.
    patient_stats : dict of str → PatientStats
        Per-patient normalization statistics computed from training data.
        Required at inference time to convert predictions back to mg/dL.
    patient_train_datasets : dict of str → GlucoseDataset
        Per-patient training datasets — available for future per-patient
        fine-tuning or leave-one-patient-out experiments.
    patient_val_datasets : dict of str → GlucoseDataset
        Per-patient validation datasets.
    """
    train_loader            : DataLoader
    val_loader              : DataLoader
    test_loaders            : Dict[str, DataLoader]
    patient_stats           : Dict[str, PatientStats]
    patient_train_datasets  : Dict[str, GlucoseDataset]
    patient_val_datasets    : Dict[str, GlucoseDataset]


def load_ohio_data(
    train_dir      : str,
    test_dir       : str,
    batch_size     : int  = 64,
    num_workers    : int  = 0,
    train_frac     : float = TRAIN_FRAC,
    patient_ids    : Optional[List[str]] = None,
    verbose        : bool = True,
) -> DataSplit:
    """
    Full data loading pipeline for OhioT1DM 2020.

    Discovers all patient XML files, tokenizes each patient, splits
    chronologically, and returns ready-to-use DataLoaders.

    Parameters
    ----------
    train_dir : str
        Path to directory containing *-ws-training.xml files.
    test_dir : str
        Path to directory containing *-ws-testing.xml files.
    batch_size : int
        Batch size for all DataLoaders.
    num_workers : int
        DataLoader worker processes. Use 0 for debugging, 4 for training.
        On macOS set to 0 to avoid multiprocessing issues with fork.
    train_frac : float
        Fraction of each patient's training sequences used for training.
        Remainder goes to validation.
    patient_ids : list of str, optional
        If provided, load only these patient IDs.
        Useful for quick experiments on a subset of patients.
        E.g., ['540', '544'] loads only those two patients.
        If None, loads all patients found in train_dir.
    verbose : bool
        Print per-patient sequence counts and verification results.

    Returns
    -------
    DataSplit containing all loaders and metadata.
    """

    # ── Discover patient IDs ──────────────────────────────────────────────────
    if patient_ids is None:
        # Extract IDs from filenames, e.g. '540-ws-training.xml' → '540'
        patient_ids = sorted([
            fname.split('-')[0]
            for fname in os.listdir(train_dir)
            if fname.endswith('-ws-training.xml')
        ])

    if verbose:
        print(f"Loading {len(patient_ids)} patients: {patient_ids}")
        print(f"Train fraction: {train_frac:.0%} / Val fraction: {1-train_frac:.0%}")
        print()

    # ── Per-patient storage ───────────────────────────────────────────────────
    patient_stats          : Dict[str, PatientStats]      = {}
    patient_train_datasets : Dict[str, GlucoseDataset]    = {}
    patient_val_datasets   : Dict[str, GlucoseDataset]    = {}
    patient_test_datasets  : Dict[str, GlucoseDataset]    = {}

    # ── Process each patient ──────────────────────────────────────────────────
    for pid in patient_ids:
        train_path = os.path.join(train_dir, f"{pid}-ws-training.xml")
        test_path  = os.path.join(test_dir,  f"{pid}-ws-testing.xml")

        if not os.path.exists(train_path):
            print(f"  WARNING: Training file not found for patient {pid}, skipping")
            continue
        if not os.path.exists(test_path):
            print(f"  WARNING: Test file not found for patient {pid}, skipping")
            continue

        if verbose:
            print(f"── Patient {pid} ──────────────────────────────────────")

        # ── Training XML ──────────────────────────────────────────────────────
        train_data = parse_ohio_xml(train_path)

        # Tokenize — stats=None means compute from this patient's training data
        all_train_seqs, stats = tokenize_patient(
            train_data, patient_id=pid, stats=None
        )

        if verbose:
            verify_tokenization(all_train_seqs)

        # Store stats — these will be used for val and test normalization
        patient_stats[pid] = stats

        # Chronological split into train and validation
        train_seqs, val_seqs = split_sequences_chronological(
            all_train_seqs, train_frac
        )

        if verbose:
            print(f"  Train sequences : {len(train_seqs)}")
            print(f"  Val sequences   : {len(val_seqs)}")

        patient_train_datasets[pid] = GlucoseDataset(train_seqs, pid, stats)
        patient_val_datasets[pid]   = GlucoseDataset(val_seqs,   pid, stats)

        # ── Test XML ──────────────────────────────────────────────────────────
        test_data = parse_ohio_xml(test_path)

        # Tokenize using training stats — critical to prevent data leakage
        # The test set is normalized using the mean/std from training data
        # exactly as it would be at real deployment time
        test_seqs, _ = tokenize_patient(
            test_data, patient_id=pid, stats=stats
        )

        if verbose:
            print(f"  Test sequences  : {len(test_seqs)}")

        patient_test_datasets[pid] = GlucoseDataset(test_seqs, pid, stats)

        if verbose:
            print()

    # ── Population DataLoaders ────────────────────────────────────────────────

    # Combine all patients into a single training dataset
    # ConcatDataset preserves order within each patient's sequences
    # Shuffling in the DataLoader then interleaves patients across batches
    population_train = ConcatDataset(list(patient_train_datasets.values()))
    population_val   = ConcatDataset(list(patient_val_datasets.values()))

    train_loader = DataLoader(
        population_train,
        batch_size  = batch_size,
        shuffle     = True,      # Shuffle across patients and time within training
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,      # Faster GPU transfer if CUDA available
        drop_last   = True,      # Drop incomplete final batch for stable training
    )

    val_loader = DataLoader(
        population_val,
        batch_size  = batch_size,
        shuffle     = False,     # Never shuffle validation — preserve temporal order
        collate_fn  = collate_fn,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    # Per-patient test loaders — one per patient, no shuffle
    # Evaluated separately to produce per-patient metrics
    test_loaders = {
        pid: DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            collate_fn  = collate_fn,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = False,
        )
        for pid, dataset in patient_test_datasets.items()
    }

    if verbose:
        print("─" * 50)
        print(f"Population training batches : {len(train_loader)}")
        print(f"Population validation batches: {len(val_loader)}")
        for pid, loader in test_loaders.items():
            print(f"Test batches (patient {pid}) : {len(loader)}")

    return DataSplit(
        train_loader           = train_loader,
        val_loader             = val_loader,
        test_loaders           = test_loaders,
        patient_stats          = patient_stats,
        patient_train_datasets = patient_train_datasets,
        patient_val_datasets   = patient_val_datasets,
    )


# ── Verification ──────────────────────────────────────────────────────────────

def verify_batch(batch: Dict, expected_batch_size: int = None):
    """
    Sanity check one batch from any DataLoader.
    Call this on the first batch of each loader before training.
    """
    print("\nBatch verification:")
    print(f"  features shape       : {batch['features'].shape}")
    print(f"  attention_mask shape : {batch['attention_mask'].shape}")
    print(f"  target shape         : {batch['target'].shape}")
    print(f"  target_raw shape     : {batch['target_raw'].shape}")
    print(f"  patient_ids          : {batch['patient_id'][:4]}...")

    # Shape checks
    B = batch['features'].shape[0]
    assert batch['features'].shape       == (B, INPUT_WINDOW, N_FEATURES), \
        f"Features shape wrong: {batch['features'].shape}"
    assert batch['attention_mask'].shape == (B, INPUT_WINDOW), \
        f"Mask shape wrong: {batch['attention_mask'].shape}"
    assert batch['target'].shape         == (B, 1), \
        f"Target shape wrong: {batch['target'].shape}"

    if expected_batch_size is not None:
        assert B == expected_batch_size, \
            f"Batch size wrong: {B} vs expected {expected_batch_size}"

    # Value checks
    assert not torch.isnan(batch['features']).any(),      "NaN in features"
    assert not torch.isnan(batch['target']).any(),        "NaN in targets"
    assert batch['attention_mask'].dtype == torch.bool,   "Mask not bool"

    # Mask sanity — at least some tokens should be valid in every sample
    tokens_valid = batch['attention_mask'].sum(dim=1)
    assert (tokens_valid > 0).all(), \
        "Some samples have zero valid tokens — check gap handling"

    print(f"  Batch size           : {B}")
    print(f"  Masked tokens/sample : {(~batch['attention_mask']).float().mean():.4f}")
    print(f"  Target range         : [{batch['target'].min():.3f}, "
          f"{batch['target'].max():.3f}] (normalized)")
    print(f"  Target raw range     : [{batch['target_raw'].min():.1f}, "
          f"{batch['target_raw'].max():.1f}] mg/dL")

    # Patient mix check — for population loader should see multiple patients
    unique_patients = set(batch['patient_id'])
    print(f"  Unique patients      : {unique_patients}")
    print("  ALL BATCH CHECKS PASSED")