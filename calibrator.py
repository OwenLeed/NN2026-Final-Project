"""
calibrator.py
=============
Conformalized Quantile Regression for guaranteed prediction interval coverage.

After training the quantile regression Transformer, this module adjusts
the raw prediction intervals to achieve mathematically guaranteed coverage
on future data.

Theory:
    Conformal prediction provides a distribution-free coverage guarantee:
    given exchangeable calibration and test data, the adjusted intervals
    will contain the true value at least (1-alpha)% of the time.

    "Exchangeable" roughly means the calibration and test data come from
    the same distribution — a reasonable assumption when both come from
    the same patient population and recording conditions.

    The guarantee is marginal — it holds on average across the full
    test distribution. It does not guarantee coverage within specific
    glycemic subranges (hypo/normal/hyper). We compute range-specific
    coverage separately as a diagnostic.

    Reference:
        Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal
        Prediction and Distribution-Free Uncertainty Quantification"

Steps:
    1. Run trained model on held-out calibration set
    2. Compute conformity scores — how much each interval missed by
    3. Find the (1-alpha)(1 + 1/n) quantile of the scores — q_hat
    4. At test time, expand intervals by q_hat in both directions

    The finite sample correction (1 + 1/n) ensures coverage holds
    exactly even for small calibration sets.
"""

import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from model  import GlucoseTransformer, TransformerConfig


# ── Calibration result ────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """
    Stores all calibration outputs for one interval level.

    Attributes
    ----------
    interval_level : str
        Human-readable label e.g. '90%'
    target_coverage : float
        Desired coverage level e.g. 0.90
    raw_coverage : float
        Empirical coverage before conformal adjustment
    adjusted_coverage : float
        Empirical coverage after conformal adjustment on calibration set
        Should be >= target_coverage by construction
    q_hat : float
        The conformal adjustment scalar in normalized glucose units
    q_hat_mgdl : Dict[str, float]
        q_hat converted to mg/dL per patient using their glucose std
        Keyed by patient_id
    n_calibration : int
        Number of calibration samples used
    lower_tau : float
        Lower quantile level e.g. 0.05 for the 90% interval
    upper_tau : float
        Upper quantile level e.g. 0.95 for the 90% interval
    range_coverage : Dict[str, float]
        Coverage broken down by glycemic range on calibration set
        Keys: 'hypo', 'normal', 'hyper' and corresponding '_n' counts
    """
    interval_level   : str
    target_coverage  : float
    raw_coverage     : float
    adjusted_coverage: float
    q_hat            : float
    q_hat_mgdl       : Dict[str, float]
    n_calibration    : int
    lower_tau        : float
    upper_tau        : float
    range_coverage   : Dict[str, float] = field(default_factory=dict)

    def summary(self):
        print(f"\n  {self.interval_level} interval "
              f"(q{int(self.lower_tau*100)}/q{int(self.upper_tau*100)}):")
        print(f"    Calibration samples : {self.n_calibration:,}")
        print(f"    Raw coverage        : {self.raw_coverage:.4f}  "
              f"(target {self.target_coverage:.2f}, "
              f"gap {self.raw_coverage - self.target_coverage:+.4f})")
        print(f"    Adjusted coverage   : {self.adjusted_coverage:.4f}  "
              f"(gap {self.adjusted_coverage - self.target_coverage:+.4f})")
        print(f"    q_hat (normalized)  : {self.q_hat:.6f}")
        if self.q_hat_mgdl:
            mean_mgdl = np.mean(list(self.q_hat_mgdl.values()))
            print(f"    q_hat (mg/dL, mean) : {mean_mgdl:.2f}")
        if self.range_coverage:
            print(f"    Range-specific coverage after adjustment:")
            for rng in ['hypo', 'normal', 'hyper']:
                cov = self.range_coverage.get(rng, float('nan'))
                n   = self.range_coverage.get(f'{rng}_n', 0)
                if not np.isnan(cov):
                    print(f"      {rng:10s}: {cov:.4f}  (n={n:,})")
                else:
                    print(f"      {rng:10s}: no samples")

# Add this dataclass after CalibrationResult in calibrator.py

@dataclass
class RangeCalibrationResult:
    """
    Stores range-specific conformal calibration results.

    One q_hat per glycemic range per interval level.
    At inference time, the appropriate q_hat is selected based on
    the patient's current observed glucose value.

    Attributes
    ----------
    interval_level : str
        e.g. '90%'
    q_hat_hypo : float
        Conformal adjustment for hypoglycemic range (< 70 mg/dL)
        in normalized glucose space
    q_hat_normal : float
        Conformal adjustment for normal range (70-180 mg/dL)
    q_hat_hyper : float
        Conformal adjustment for hyperglycemic range (> 180 mg/dL)
    n_hypo, n_normal, n_hyper : int
        Calibration sample counts per range
    coverage_hypo, coverage_normal, coverage_hyper : float
        Empirical coverage per range after range-specific adjustment
    lower_tau : float
    upper_tau : float
    """
    interval_level   : str
    q_hat_hypo       : float
    q_hat_normal     : float
    q_hat_hyper      : float
    n_hypo           : int
    n_normal         : int
    n_hyper          : int
    coverage_hypo    : float
    coverage_normal  : float
    coverage_hyper   : float
    target_coverage  : float
    lower_tau        : float
    upper_tau        : float

    def summary(self):
        print(f"\n  {self.interval_level} range-specific calibration:")
        print(f"  {'Range':10s}  {'n':>7}  {'q_hat':>10}  "
              f"{'Coverage':>10}  {'Target':>8}  {'Gap':>8}")
        print(f"  {'-'*58}")
        rows = [
            ('hypo',   self.n_hypo,   self.q_hat_hypo,
             self.coverage_hypo),
            ('normal', self.n_normal, self.q_hat_normal,
             self.coverage_normal),
            ('hyper',  self.n_hyper,  self.q_hat_hyper,
             self.coverage_hyper),
        ]
        for name, n, q, cov in rows:
            gap = cov - self.target_coverage
            print(f"  {name:10s}  {n:>7,}  {q:>10.6f}  "
                  f"{cov:>10.4f}  {self.target_coverage:>8.2f}  "
                  f"{gap:>+8.4f}")


# ── Conformal calibrator ──────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Applies Conformalized Quantile Regression to a trained
    GlucoseTransformer.

    The calibration set must be:
        - Held out from training (never seen by the model during training)
        - Representative of the test distribution
        - Large enough for reliable quantile estimation
          (n >= 100 is a practical minimum; you have ~18k which is excellent)

    Parameters
    ----------
    model : GlucoseTransformer
        Trained model — weights are not modified by calibration.
    model_config : TransformerConfig
        Model configuration containing quantile levels.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model        : GlucoseTransformer,
        model_config : TransformerConfig,
        device       : torch.device,
    ):
        self.model        = model
        self.model_config = model_config
        self.device       = device
        self.quantiles    = model_config.quantiles
        self.results      : Dict[str, CalibrationResult] = {}

    # ── Collect predictions on calibration set ────────────────────────────────

    @torch.no_grad()
    def _collect_predictions(
        self,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Run the model over the calibration set and collect all outputs.

        Returns
        -------
        predictions   : (n, n_quantiles) — normalized quantile predictions
        targets       : (n,)             — normalized true glucose values
        targets_raw   : (n,)             — true glucose in mg/dL
        patient_ids   : list of str      — patient id per sample
        """
        self.model.eval()

        all_preds       = []
        all_targets     = []
        all_targets_raw = []
        all_pids        = []

        for batch in loader:
            features = batch['features'].to(self.device)
            mask     = batch['attention_mask'].to(self.device)

            preds = self.model(features, mask).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch['target'].numpy().flatten())
            all_targets_raw.append(batch['target_raw'].numpy().flatten())
            all_pids.extend(batch['patient_id'])

        return (
            np.concatenate(all_preds,       axis=0),
            np.concatenate(all_targets,     axis=0),
            np.concatenate(all_targets_raw, axis=0),
            all_pids,
        )

    # ── Compute conformity scores ─────────────────────────────────────────────

    def _conformity_scores(
        self,
        predictions : np.ndarray,
        targets     : np.ndarray,
        lower_idx   : int,
        upper_idx   : int,
    ) -> np.ndarray:
        """
        Compute conformity scores for one interval.

        The conformity score measures how much we need to expand the
        predicted interval to capture the true value.

        For a sample where the true value is y and predicted interval [l, u]:
            score = max(l - y, y - u)

        Interpretation:
            score < 0 — true value is INSIDE the interval (good)
            score > 0 — true value is OUTSIDE the interval (bad)
                        value tells you exactly how much expansion is needed

        Parameters
        ----------
        predictions : (n, n_quantiles)
        targets     : (n,)             normalized true values
        lower_idx   : column index of lower quantile
        upper_idx   : column index of upper quantile
        """
        lower  = predictions[:, lower_idx]
        upper  = predictions[:, upper_idx]
        scores = np.maximum(lower - targets, targets - upper)
        return scores

    # ── Range-specific coverage ───────────────────────────────────────────────

    def _range_specific_coverage(
        self,
        lower        : np.ndarray,
        upper        : np.ndarray,
        targets_raw  : np.ndarray,
        targets_norm : np.ndarray,
    ) -> Dict[str, float]:
        """
        Coverage broken down by glycemic range.

        This is the key diagnostic that published work almost never reports.
        A model can achieve 90% aggregate coverage while badly miscovering
        in the hypoglycemic range — which is the clinically critical case.

        Parameters
        ----------
        lower        : (n,) adjusted lower bounds in normalized space
        upper        : (n,) adjusted upper bounds in normalized space
        targets_raw  : (n,) true glucose values in mg/dL
                       used only for range classification
        targets_norm : (n,) true glucose values in normalized space
                       used for the actual coverage computation

        Ranges:
            Hypo   : < 70 mg/dL
            Normal : 70–180 mg/dL
            Hyper  : > 180 mg/dL
        """
        ranges = {
            'hypo'  : targets_raw < 70,
            'normal': (targets_raw >= 70) & (targets_raw <= 180),
            'hyper' : targets_raw > 180,
        }

        results = {}
        for name, mask in ranges.items():
            if mask.sum() > 0:
                # Coverage = fraction of samples in this range where
                # the true normalized value falls within the adjusted interval
                in_interval = (
                    (targets_norm[mask] >= lower[mask]) &
                    (targets_norm[mask] <= upper[mask])
                )
                results[name]          = float(in_interval.mean())
                results[f'{name}_n']   = int(mask.sum())
            else:
                results[name]          = float('nan')
                results[f'{name}_n']   = 0

        return results

    # ── Main calibration method ───────────────────────────────────────────────

    def calibrate(
        self,
        cal_loader   : DataLoader,
        patient_stats: Dict,
        alpha_levels : Optional[List[float]] = None,
    ) -> Dict[str, CalibrationResult]:
        """
        Run full conformal calibration.

        Parameters
        ----------
        cal_loader : DataLoader
            DataLoader for the calibration set.
            Use the validation set from DataSplit — it was never used
            for gradient updates, only for early stopping decisions.
            With 18,848 sequences this is a very well-powered calibration.
        patient_stats : dict
            Per-patient normalization stats for mg/dL conversion.
        alpha_levels : list of float, optional
            Miscoverage rates to calibrate. E.g., [0.10, 0.50] gives
            90% and 50% intervals.
            Defaults to all symmetric pairs in the model's quantiles.

        Returns
        -------
        Dict mapping interval label (e.g. '90%') to CalibrationResult.
        """
        print("Running conformal calibration...")
        print(f"  Quantiles : {self.quantiles}")

        # Collect all predictions on calibration set
        predictions, targets, targets_raw, patient_ids = (
            self._collect_predictions(cal_loader)
        )
        n = len(targets)
        print(f"  Calibration samples: {n:,}")

        # Determine interval pairs from quantiles if not specified
        if alpha_levels is None:
            alpha_levels = []
            q = self.quantiles
            for i in range(len(q) // 2):
                alpha = round(1.0 - (q[-(i+1)] - q[i]), 10)
                alpha_levels.append(alpha)

        print(f"  Calibrating intervals: "
              f"{[f'{round((1-a)*100)}%' for a in alpha_levels]}")
        print()

        for alpha in alpha_levels:
            target_coverage = round(1.0 - alpha, 10)
            label           = f"{round(target_coverage * 100)}%"

            # Find quantile indices for this interval
            lower_tau = round(alpha / 2, 10)
            upper_tau = round(1.0 - alpha / 2, 10)

            # Match to nearest quantile in model's quantile list
            lower_idx = min(
                range(len(self.quantiles)),
                key=lambda i: abs(self.quantiles[i] - lower_tau)
            )
            upper_idx = min(
                range(len(self.quantiles)),
                key=lambda i: abs(self.quantiles[i] - upper_tau)
            )

            # Raw coverage before adjustment
            lower_raw    = predictions[:, lower_idx]
            upper_raw    = predictions[:, upper_idx]
            raw_coverage = float(np.mean(
                (targets >= lower_raw) & (targets <= upper_raw)
            ))

            # Conformity scores
            scores = self._conformity_scores(
                predictions, targets, lower_idx, upper_idx
            )

            # q_hat — the conformal adjustment
            # Finite sample correction: (1-alpha)(1 + 1/n) ensures
            # the coverage guarantee holds exactly, not just asymptotically
            correction_level = min((1.0 - alpha) * (1.0 + 1.0 / n), 1.0)
            q_hat            = float(np.quantile(scores, correction_level))

            # Adjusted bounds on calibration set
            lower_adj    = lower_raw - q_hat
            upper_adj    = upper_raw + q_hat

            # Adjusted coverage on calibration set
            adj_coverage = float(np.mean(
                (targets >= lower_adj) & (targets <= upper_adj)
            ))

            # Convert q_hat to mg/dL per patient
            q_hat_mgdl = {}
            for pid, stats in patient_stats.items():
                q_hat_mgdl[pid] = q_hat * stats.glucose_std

            # Range-specific coverage after adjustment
            # Pass both raw (for range classification) and
            # normalized (for coverage computation) targets
            range_coverage = self._range_specific_coverage(
                lower_adj, upper_adj, targets_raw, targets
            )

            result = CalibrationResult(
                interval_level    = label,
                target_coverage   = target_coverage,
                raw_coverage      = raw_coverage,
                adjusted_coverage = adj_coverage,
                q_hat             = q_hat,
                q_hat_mgdl        = q_hat_mgdl,
                n_calibration     = n,
                lower_tau         = self.quantiles[lower_idx],
                upper_tau         = self.quantiles[upper_idx],
                range_coverage    = range_coverage,
            )

            self.results[label] = result
            result.summary()

        return self.results

    # ── Apply calibration to new predictions ──────────────────────────────────

    def adjust(
        self,
        predictions    : np.ndarray,
        interval_level : str = '90%',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply conformal adjustment to raw model predictions.

        Parameters
        ----------
        predictions : (n, n_quantiles)
            Raw model output in normalized glucose space.
        interval_level : str
            Which interval to return, e.g. '90%' or '50%'.

        Returns
        -------
        point_pred : (n,) — median prediction (normalized)
        lower      : (n,) — adjusted lower bound (normalized)
        upper      : (n,) — adjusted upper bound (normalized)
        """
        assert interval_level in self.results, (
            f"Interval '{interval_level}' not calibrated. "
            f"Available: {list(self.results.keys())}"
        )

        result     = self.results[interval_level]
        q_hat      = result.q_hat
        lower_idx  = self.quantiles.index(result.lower_tau)
        upper_idx  = self.quantiles.index(result.upper_tau)
        median_idx = self.quantiles.index(0.50)

        point_pred = predictions[:, median_idx]
        lower      = predictions[:, lower_idx]  - q_hat
        upper      = predictions[:, upper_idx]  + q_hat

        return point_pred, lower, upper

    def adjust_to_mgdl(
        self,
        predictions   : np.ndarray,
        patient_id    : str,
        patient_stats : Dict,
        interval_level: str = '90%',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply conformal adjustment and convert output to mg/dL.

        Parameters
        ----------
        predictions   : (n, n_quantiles) normalized
        patient_id    : str
        patient_stats : dict of PatientStats
        interval_level: str

        Returns
        -------
        point_pred, lower, upper — all in mg/dL
        """
        point_norm, lower_norm, upper_norm = self.adjust(
            predictions, interval_level
        )

        stats = patient_stats[patient_id]
        mu, sigma = stats.glucose_mean, stats.glucose_std

        point_mgdl = point_norm * sigma + mu
        lower_mgdl = lower_norm * sigma + mu
        upper_mgdl = upper_norm * sigma + mu

        return point_mgdl, lower_mgdl, upper_mgdl
    
    # --------------------------- Range-specific calibration and adjustment ---------------------------

    def calibrate_by_range(
        self,
        cal_loader   : DataLoader,
        alpha_levels : Optional[List[float]] = None,
    ) -> Dict[str, RangeCalibrationResult]:
        """
        Compute separate conformal adjustments per glycemic range.

        This addresses the core limitation of global conformal calibration:
        a single q_hat achieves aggregate coverage but systematically
        undercoveres hypoglycemia because hypoglycemic samples are rare
        and have different uncertainty characteristics.

        By computing q_hat separately within each range, we guarantee
        (1-alpha) coverage within each range independently, subject to
        the exchangeability assumption holding within each subgroup.

        Clinical interpretation:
            At inference time, if the patient's most recent observed
            glucose is < 70 mg/dL, use q_hat_hypo to widen the interval.
            If >= 70 and <= 180, use q_hat_normal.
            If > 180, use q_hat_hyper.

        Parameters
        ----------
        cal_loader : DataLoader
            Same calibration loader used for global calibration.
        alpha_levels : list of float, optional
            Miscoverage rates. Defaults to symmetric pairs from quantiles.

        Returns
        -------
        Dict mapping interval label to RangeCalibrationResult.
        """
        print("\nRunning range-specific conformal calibration...")

        # Collect all predictions and targets
        predictions, targets, targets_raw, patient_ids = (
            self._collect_predictions(cal_loader)
        )
        n = len(targets)

        # Define glycemic ranges using raw mg/dL values
        range_masks = {
            'hypo'  : targets_raw < 70,
            'normal': (targets_raw >= 70) & (targets_raw <= 180),
            'hyper' : targets_raw > 180,
        }

        for name, mask in range_masks.items():
            print(f"  {name:10s}: {mask.sum():,} samples "
                f"({mask.mean()*100:.1f}%)")

        # Determine alpha levels
        if alpha_levels is None:
            alpha_levels = []
            q = self.quantiles
            for i in range(len(q) // 2):
                alpha = round(1.0 - (q[-(i+1)] - q[i]), 10)
                alpha_levels.append(alpha)

        range_results = {}

        for alpha in alpha_levels:
            target_coverage = round(1.0 - alpha, 10)
            label           = f"{round(target_coverage * 100)}%"

            # Find quantile indices
            lower_tau = round(alpha / 2, 10)
            upper_tau = round(1.0 - alpha / 2, 10)
            lower_idx = min(
                range(len(self.quantiles)),
                key=lambda i: abs(self.quantiles[i] - lower_tau)
            )
            upper_idx = min(
                range(len(self.quantiles)),
                key=lambda i: abs(self.quantiles[i] - upper_tau)
            )

            q_hats   = {}
            coverage = {}
            counts   = {}

            for range_name, mask in range_masks.items():
                n_range = mask.sum()
                counts[range_name] = int(n_range)

                if n_range < 10:
                    # Too few samples for reliable calibration
                    # Fall back to global q_hat for this range
                    print(f"  WARNING: {range_name} has only {n_range} samples "
                        f"— falling back to global q_hat")
                    q_hats[range_name] = self.results[label].q_hat \
                        if label in self.results else 0.0
                    coverage[range_name] = float('nan')
                    continue

                # Conformity scores within this range only
                scores_range = self._conformity_scores(
                    predictions[mask], targets[mask],
                    lower_idx, upper_idx
                )

                # Range-specific q_hat with finite sample correction
                correction   = min(
                    (1.0 - alpha) * (1.0 + 1.0 / n_range), 1.0
                )
                q_hat_range  = float(np.quantile(scores_range, correction))
                q_hats[range_name] = q_hat_range

                # Empirical coverage within this range after adjustment
                lower_adj = predictions[mask, lower_idx] - q_hat_range
                upper_adj = predictions[mask, upper_idx] + q_hat_range
                cov       = float(np.mean(
                    (targets[mask] >= lower_adj) &
                    (targets[mask] <= upper_adj)
                ))
                coverage[range_name] = cov

            result = RangeCalibrationResult(
                interval_level  = label,
                q_hat_hypo      = q_hats['hypo'],
                q_hat_normal    = q_hats['normal'],
                q_hat_hyper     = q_hats['hyper'],
                n_hypo          = counts['hypo'],
                n_normal        = counts['normal'],
                n_hyper         = counts['hyper'],
                coverage_hypo   = coverage.get('hypo', float('nan')),
                coverage_normal = coverage.get('normal', float('nan')),
                coverage_hyper  = coverage.get('hyper', float('nan')),
                target_coverage = target_coverage,
                lower_tau       = self.quantiles[lower_idx],
                upper_tau       = self.quantiles[upper_idx],
            )

            range_results[label] = result
            result.summary()

        self.range_results = range_results
        return range_results


    def adjust_by_range(
        self,
        predictions      : np.ndarray,
        current_glucose  : np.ndarray,
        interval_level   : str = '90%',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply range-specific conformal adjustment.

        Selects q_hat based on each sample's most recent observed glucose
        value — the last reading in the input window, which is the best
        proxy for the patient's current glycemic state.

        Parameters
        ----------
        predictions     : (n, n_quantiles) normalized model output
        current_glucose : (n,) most recent observed glucose in mg/dL
                        Use the last valid token's raw glucose value
        interval_level  : str e.g. '90%'

        Returns
        -------
        point_pred : (n,) median prediction normalized
        lower      : (n,) range-adjusted lower bound normalized
        upper      : (n,) range-adjusted upper bound normalized
        """
        assert hasattr(self, 'range_results'), \
            "Run calibrate_by_range() before calling adjust_by_range()"
        assert interval_level in self.range_results, \
            f"Interval '{interval_level}' not range-calibrated."

        result     = self.range_results[interval_level]
        lower_idx  = self.quantiles.index(result.lower_tau)
        upper_idx  = self.quantiles.index(result.upper_tau)
        median_idx = self.quantiles.index(0.50)

        # Select q_hat per sample based on current glucose
        q_hats = np.where(
            current_glucose < 70,
            result.q_hat_hypo,
            np.where(
                current_glucose <= 180,
                result.q_hat_normal,
                result.q_hat_hyper,
            )
        )

        point_pred = predictions[:, median_idx]
        lower      = predictions[:, lower_idx]  - q_hats
        upper      = predictions[:, upper_idx]  + q_hats

        return point_pred, lower, upper

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save calibration results to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Calibration results saved to {path}")

    def load(self, path: str):
        """Load calibration results from disk."""
        with open(path, 'rb') as f:
            self.results = pickle.load(f)
        print(f"Calibration results loaded from {path}")

