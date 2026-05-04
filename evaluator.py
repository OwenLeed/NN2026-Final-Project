"""
evaluator.py
============
Final evaluation of the calibrated GlucoseTransformer on the held-out
test set.

Computes:
    1. Point prediction metrics (RMSE, MAE, MAPE) — overall and per patient
    2. Coverage metrics — global and range-specific, before and after
       conformal calibration
    3. Interval width metrics — how wide are the prediction bands
    4. Clarke Error Grid analysis — clinical safety metric
    5. Per-patient breakdown — identifies which patients are harder to predict

All metrics are computed in mg/dL for clinical interpretability.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from model      import GlucoseTransformer, TransformerConfig
from calibrator import ConformalCalibrator


# ── Clarke Error Grid ─────────────────────────────────────────────────────────

def clarke_error_grid(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Clarke Error Grid zone distribution.

    The Clarke Error Grid is the standard clinical safety metric for
    glucose prediction. It divides the (reference, predicted) space
    into 5 zones:

        Zone A — clinically accurate (within 20% of reference)
        Zone B — benign errors (outside 20% but no dangerous treatment)
        Zone C — overcorrection errors (would cause unnecessary treatment)
        Zone D — dangerous failure to detect (missed hypo/hyper)
        Zone E — erroneous treatment errors (most dangerous)

    For a clinically acceptable system, Zone A+B should exceed 99%.
    Zone D and E should be as close to 0% as possible.

    Parameters
    ----------
    ref  : (n,) true glucose values in mg/dL
    pred : (n,) predicted glucose values in mg/dL

    Returns
    -------
    Dict with zone percentages and counts.
    """
    n      = len(ref)
    zones  = np.zeros(n, dtype=int)

    for i in range(n):
        r = ref[i]
        p = pred[i]

        # Zone E
        if (r <= 70 and p >= 180) or (r >= 180 and p <= 70):
            zones[i] = 4
        # Zone D
        elif (r >= 240 and p <= 70) or (r <= 70 and p >= 180):
            zones[i] = 3
        # Zone C
        elif ((r >= 70 and r <= 290) and
              (p >= r + 110)) or \
             ((r >= 130 and r <= 180) and
              (p <= (7/5) * r - 182)):
            zones[i] = 2
        # Zone A
        elif (abs(p - r) <= 0.2 * r) or \
             (r <= 58.33 and p <= 70):
            zones[i] = 0
        # Zone B (everything else)
        else:
            zones[i] = 1

    counts = {
        'A': int((zones == 0).sum()),
        'B': int((zones == 1).sum()),
        'C': int((zones == 2).sum()),
        'D': int((zones == 3).sum()),
        'E': int((zones == 4).sum()),
    }
    pcts = {k: v / n * 100 for k, v in counts.items()}

    return {
        'counts'  : counts,
        'pcts'    : pcts,
        'A+B'     : pcts['A'] + pcts['B'],
        'n'       : n,
    }


# ── Metric computation ────────────────────────────────────────────────────────

def compute_point_metrics(
    preds_mgdl  : np.ndarray,
    targets_mgdl: np.ndarray,
) -> Dict[str, float]:
    """
    Standard point prediction metrics in mg/dL.

    RMSE  — root mean squared error, penalizes large errors heavily
    MAE   — mean absolute error, more interpretable clinically
    MAPE  — mean absolute percentage error
    MBE   — mean bias error (signed), reveals systematic over/under-prediction
    """
    errors = preds_mgdl - targets_mgdl
    abs_e  = np.abs(errors)

    return {
        'RMSE' : float(np.sqrt(np.mean(errors ** 2))),
        'MAE'  : float(np.mean(abs_e)),
        'MAPE' : float(np.mean(abs_e / np.maximum(targets_mgdl, 1.0)) * 100),
        'MBE'  : float(np.mean(errors)),
    }


def compute_coverage_metrics(
    lower_mgdl   : np.ndarray,
    upper_mgdl   : np.ndarray,
    targets_mgdl : np.ndarray,
    target_cov   : float,
) -> Dict[str, float]:
    """
    Coverage and interval width metrics for one confidence level.

    Coverage  — fraction of true values inside the interval
    Width     — mean interval width in mg/dL
    Efficiency— coverage / width ratio (higher = more informative)
    """
    in_interval = (
        (targets_mgdl >= lower_mgdl) &
        (targets_mgdl <= upper_mgdl)
    )
    coverage = float(in_interval.mean())
    width    = float(np.mean(upper_mgdl - lower_mgdl))

    return {
        'coverage'   : coverage,
        'width_mean' : width,
        'width_std'  : float(np.std(upper_mgdl - lower_mgdl)),
        'gap'        : coverage - target_cov,
    }


def compute_range_metrics(
    preds_mgdl   : np.ndarray,
    targets_mgdl : np.ndarray,
    lower_mgdl   : np.ndarray,
    upper_mgdl   : np.ndarray,
    target_cov   : float,
) -> Dict[str, Dict]:
    """
    Point prediction and coverage metrics broken down by glycemic range.
    """
    ranges = {
        'hypo'  : targets_mgdl < 70,
        'normal': (targets_mgdl >= 70) & (targets_mgdl <= 180),
        'hyper' : targets_mgdl > 180,
    }

    results = {}
    for name, mask in ranges.items():
        if mask.sum() == 0:
            results[name] = {'n': 0}
            continue

        point = compute_point_metrics(
            preds_mgdl[mask], targets_mgdl[mask]
        )
        cov = compute_coverage_metrics(
            lower_mgdl[mask], upper_mgdl[mask],
            targets_mgdl[mask], target_cov
        )
        results[name] = {
            'n'       : int(mask.sum()),
            **point,
            **cov,
        }

    return results


# ── Main evaluator ────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs full evaluation of a calibrated GlucoseTransformer on test data.

    Parameters
    ----------
    model       : trained GlucoseTransformer
    calibrator  : fitted ConformalCalibrator with range results
    model_config: TransformerConfig
    device      : torch.device
    """

    def __init__(
        self,
        model        : GlucoseTransformer,
        calibrator   : ConformalCalibrator,
        model_config : TransformerConfig,
        device       : torch.device,
    ):
        self.model        = model
        self.calibrator   = calibrator
        self.model_config = model_config
        self.device       = device
        self.quantiles    = model_config.quantiles

    @torch.no_grad()
    def _collect_test_predictions(
        self,
        loader       : DataLoader,
        patient_id   : str,
        patient_stats: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Run model on one patient's test loader and collect all outputs,
        converting everything to mg/dL.
        """
        self.model.eval()

        all_preds_norm = []
        all_targets    = []
        all_targets_raw= []
        all_current_g  = []

        stats = patient_stats[patient_id]
        mu, sigma = stats.glucose_mean, stats.glucose_std

        for batch in loader:
            features = batch['features'].to(self.device)
            mask     = batch['attention_mask'].to(self.device)

            preds_norm = self.model(features, mask).cpu().numpy()
            all_preds_norm.append(preds_norm)
            all_targets.append(batch['target'].numpy().flatten())
            all_targets_raw.append(batch['target_raw'].numpy().flatten())

            # Current glucose = last valid token's glucose feature (f0)
            # f0 is normalized glucose — convert back to mg/dL
            # Shape: (batch, seq_len, n_features) → last timestep, feature 0
            last_glucose_norm = batch['features'][:, -1, 0].numpy()
            last_glucose_mgdl = last_glucose_norm * sigma + mu
            all_current_g.append(last_glucose_mgdl)

        preds_norm   = np.concatenate(all_preds_norm,  axis=0)
        targets_norm = np.concatenate(all_targets,     axis=0)
        targets_raw  = np.concatenate(all_targets_raw, axis=0)
        current_g    = np.concatenate(all_current_g,   axis=0)

        # Sort quantile outputs to prevent crossing
        preds_norm = np.sort(preds_norm, axis=1)

        # Median point prediction in mg/dL
        median_idx   = self.quantiles.index(0.50)
        pred_mgdl    = preds_norm[:, median_idx] * sigma + mu

        # Global calibrated intervals in mg/dL
        p_g, l_g, u_g = self.calibrator.adjust(preds_norm, '90%')
        lower_global_90 = l_g * sigma + mu
        upper_global_90 = u_g * sigma + mu

        p_g50, l_g50, u_g50 = self.calibrator.adjust(preds_norm, '50%')
        lower_global_50 = l_g50 * sigma + mu
        upper_global_50 = u_g50 * sigma + mu

        # Range-specific calibrated intervals in mg/dL
        p_r, l_r, u_r = self.calibrator.adjust_by_range(
            preds_norm, current_g, '90%'
        )
        lower_range_90 = l_r * sigma + mu
        upper_range_90 = u_r * sigma + mu

        p_r50, l_r50, u_r50 = self.calibrator.adjust_by_range(
            preds_norm, current_g, '50%'
        )
        lower_range_50 = l_r50 * sigma + mu
        upper_range_50 = u_r50 * sigma + mu

        return {
            'pred_mgdl'        : pred_mgdl,
            'targets_mgdl'     : targets_raw,
            'current_g'        : current_g,
            'lower_global_90'  : lower_global_90,
            'upper_global_90'  : upper_global_90,
            'lower_global_50'  : lower_global_50,
            'upper_global_50'  : upper_global_50,
            'lower_range_90'   : lower_range_90,
            'upper_range_90'   : upper_range_90,
            'lower_range_50'   : lower_range_50,
            'upper_range_50'   : upper_range_50,
        }

    def evaluate(
        self,
        test_loaders : Dict[str, DataLoader],
        patient_stats: Dict,
    ) -> pd.DataFrame:

        all_rows    = []
        all_preds   = []
        all_targets = []
        all_l_g90   = []
        all_u_g90   = []
        all_l_r90   = []
        all_u_r90   = []
        all_l_g50   = []   # ← added
        all_u_g50   = []   # ← added
        all_l_r50   = []   # ← added
        all_u_r50   = []   # ← added

        print("Evaluating on test set...")
        print()

        for pid, loader in test_loaders.items():
            outputs = self._collect_test_predictions(
                loader, pid, patient_stats
            )

            p   = outputs['pred_mgdl']
            t   = outputs['targets_mgdl']
            lg  = outputs['lower_global_90']
            ug  = outputs['upper_global_90']
            lr  = outputs['lower_range_90']
            ur  = outputs['upper_range_90']
            lg5 = outputs['lower_global_50']
            ug5 = outputs['upper_global_50']
            lr5 = outputs['lower_range_50']
            ur5 = outputs['upper_range_50']

            # Accumulate for aggregate metrics
            all_preds.append(p)
            all_targets.append(t)
            all_l_g90.append(lg)
            all_u_g90.append(ug)
            all_l_r90.append(lr)
            all_u_r90.append(ur)
            all_l_g50.append(lg5)   # ← added
            all_u_g50.append(ug5)   # ← added
            all_l_r50.append(lr5)   # ← added
            all_u_r50.append(ur5)   # ← added

            # Point metrics
            pt  = compute_point_metrics(p, t)
            ceg = clarke_error_grid(t, p)

            # Coverage metrics
            gc90 = compute_coverage_metrics(lg,  ug,  t, 0.90)
            gc50 = compute_coverage_metrics(lg5, ug5, t, 0.50)
            rc90 = compute_coverage_metrics(lr,  ur,  t, 0.90)
            rc50 = compute_coverage_metrics(lr5, ur5, t, 0.50)
            rb   = compute_range_metrics(p, t, lr, ur, 0.90)

            row = {
                'Patient'              : pid,
                'N'                    : len(t),
                'RMSE'                 : round(pt['RMSE'],  2),
                'MAE'                  : round(pt['MAE'],   2),
                'MBE'                  : round(pt['MBE'],   2),
                'CEG A+B (%)'          : round(ceg['A+B'],  2),
                'CEG A (%)'            : round(ceg['pcts']['A'], 2),
                'Global Cov 90%'       : round(gc90['coverage'],   4),
                'Global Width 90%'     : round(gc90['width_mean'], 2),
                'Global Cov 50%'       : round(gc50['coverage'],   4),
                'Global Width 50%'     : round(gc50['width_mean'], 2),
                'Range Cov 90%'        : round(rc90['coverage'],   4),
                'Range Width 90%'      : round(rc90['width_mean'], 2),
                'Range Cov 50%'        : round(rc50['coverage'],   4),
                'Range Width 50%'      : round(rc50['width_mean'], 2),
                'Hypo N'               : rb['hypo'].get('n', 0),
                'Hypo RMSE'            : round(rb['hypo'].get('RMSE', float('nan')), 2),
                'Hypo Cov 90%'         : round(rb['hypo'].get('coverage', float('nan')), 4),
                'Hypo Width 90%'       : round(rb['hypo'].get('width_mean', float('nan')), 2),
                'Normal Cov 90%'       : round(rb['normal'].get('coverage', float('nan')), 4),
                'Hyper Cov 90%'        : round(rb['hyper'].get('coverage', float('nan')), 4),
            }
            all_rows.append(row)

            # Per-patient print
            print(f"── Patient {pid} (n={len(t):,}) "
                  f"─────────────────────────────")
            print(f"  Point:   RMSE={pt['RMSE']:.2f}  "
                  f"MAE={pt['MAE']:.2f}  "
                  f"MBE={pt['MBE']:+.2f} mg/dL")
            print(f"  CEG:     A={ceg['pcts']['A']:.1f}%  "
                  f"A+B={ceg['A+B']:.1f}%")
            print(f"  Global conformal:")
            print(f"    90%: cov={gc90['coverage']:.4f}  "
                  f"width={gc90['width_mean']:.1f} mg/dL")
            print(f"    50%: cov={gc50['coverage']:.4f}  "
                  f"width={gc50['width_mean']:.1f} mg/dL")
            print(f"  Range-specific conformal:")
            print(f"    90%: cov={rc90['coverage']:.4f}  "
                  f"width={rc90['width_mean']:.1f} mg/dL")
            print(f"    50%: cov={rc50['coverage']:.4f}  "
                  f"width={rc50['width_mean']:.1f} mg/dL")
            for rng in ['hypo', 'normal', 'hyper']:
                rdata = rb[rng]
                if rdata.get('n', 0) > 0:
                    print(f"    {rng:8s}: "
                          f"n={rdata['n']:4d}  "
                          f"RMSE={rdata['RMSE']:.2f}  "
                          f"cov={rdata['coverage']:.4f}  "
                          f"width={rdata['width_mean']:.1f}")
            print()

        # ── Aggregate ─────────────────────────────────────────────────────────
        all_preds   = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_l_g90   = np.concatenate(all_l_g90)
        all_u_g90   = np.concatenate(all_u_g90)
        all_l_r90   = np.concatenate(all_l_r90)
        all_u_r90   = np.concatenate(all_u_r90)
        all_l_g50   = np.concatenate(all_l_g50)
        all_u_g50   = np.concatenate(all_u_g50)
        all_l_r50   = np.concatenate(all_l_r50)
        all_u_r50   = np.concatenate(all_u_r50)

        pt_all   = compute_point_metrics(all_preds, all_targets)
        ceg_all  = clarke_error_grid(all_targets, all_preds)
        gc90_all = compute_coverage_metrics(all_l_g90, all_u_g90, all_targets, 0.90)
        gc50_all = compute_coverage_metrics(all_l_g50, all_u_g50, all_targets, 0.50)
        rc90_all = compute_coverage_metrics(all_l_r90, all_u_r90, all_targets, 0.90)
        rc50_all = compute_coverage_metrics(all_l_r50, all_u_r50, all_targets, 0.50)
        rb_all   = compute_range_metrics(
            all_preds, all_targets, all_l_r90, all_u_r90, 0.90
        )
        rb50_all = compute_range_metrics(
            all_preds, all_targets, all_l_r50, all_u_r50, 0.50
        )

        agg_row = {
            'Patient'              : 'ALL',
            'N'                    : len(all_targets),
            'RMSE'                 : round(pt_all['RMSE'],  2),
            'MAE'                  : round(pt_all['MAE'],   2),
            'MBE'                  : round(pt_all['MBE'],   2),
            'CEG A+B (%)'          : round(ceg_all['A+B'],  2),
            'CEG A (%)'            : round(ceg_all['pcts']['A'], 2),
            'Global Cov 90%'       : round(gc90_all['coverage'],   4),
            'Global Width 90%'     : round(gc90_all['width_mean'], 2),
            'Global Cov 50%'       : round(gc50_all['coverage'],   4),
            'Global Width 50%'     : round(gc50_all['width_mean'], 2),
            'Range Cov 90%'        : round(rc90_all['coverage'],   4),
            'Range Width 90%'      : round(rc90_all['width_mean'], 2),
            'Range Cov 50%'        : round(rc50_all['coverage'],   4),
            'Range Width 50%'      : round(rc50_all['width_mean'], 2),
            'Hypo N'               : rb_all['hypo'].get('n', 0),
            'Hypo RMSE'            : round(rb_all['hypo'].get('RMSE', float('nan')), 2),
            'Hypo Cov 90%'         : round(rb_all['hypo'].get('coverage', float('nan')), 4),
            'Hypo Width 90%'       : round(rb_all['hypo'].get('width_mean', float('nan')), 2),
            'Normal Cov 90%'       : round(rb_all['normal'].get('coverage', float('nan')), 4),
            'Hyper Cov 90%'        : round(rb_all['hyper'].get('coverage', float('nan')), 4),
        }
        all_rows.append(agg_row)

        # ── Print aggregate summary ───────────────────────────────────────────
        print("── AGGREGATE (all patients) "
              "─────────────────────────────────")
        print(f"  N             : {len(all_targets):,}")
        print(f"  RMSE          : {pt_all['RMSE']:.2f} mg/dL")
        print(f"  MAE           : {pt_all['MAE']:.2f} mg/dL")
        print(f"  MBE           : {pt_all['MBE']:+.2f} mg/dL")
        print(f"  CEG A+B       : {ceg_all['A+B']:.2f}%")
        print()
        print(f"  Global conformal:")
        print(f"    90%: cov={gc90_all['coverage']:.4f}  "
              f"width={gc90_all['width_mean']:.1f} mg/dL")
        print(f"    50%: cov={gc50_all['coverage']:.4f}  "
              f"width={gc50_all['width_mean']:.1f} mg/dL")
        print()
        print(f"  Range-specific conformal:")
        print(f"    90%: cov={rc90_all['coverage']:.4f}  "
              f"width={rc90_all['width_mean']:.1f} mg/dL")
        print(f"    50%: cov={rc50_all['coverage']:.4f}  "
              f"width={rc50_all['width_mean']:.1f} mg/dL")
        for rng in ['hypo', 'normal', 'hyper']:
            rdata = rb_all[rng]
            if rdata.get('n', 0) > 0:
                print(f"    {rng:8s}: "
                      f"n={rdata['n']:5d}  "
                      f"RMSE={rdata['RMSE']:.2f}  "
                      f"cov={rdata['coverage']:.4f}  "
                      f"width={rdata['width_mean']:.1f}")

        # ── Results DataFrame ─────────────────────────────────────────────────
        df = pd.DataFrame(all_rows)
        df = df.set_index('Patient')

        print(f"\n{'='*60}")
        print("RESULTS TABLE")
        print(f"{'='*60}")
        print(df.to_string())

        return df