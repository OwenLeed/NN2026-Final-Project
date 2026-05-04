"""
loss.py
=======
Pinball loss for quantile regression training.

The pinball loss is what causes each output head to learn a specific
quantile of the conditional glucose distribution rather than the mean.
It is the only training signal the model receives — getting it right
is critical.
"""

import torch
import torch.nn as nn
from typing import List


class PinballLoss(nn.Module):
    """
    Pinball loss (also called quantile loss) for training quantile
    regression models.

    For a single prediction at quantile τ:

        L(y, ŷ, τ) = τ × (y - ŷ)        if y >= ŷ  (model undershot)
                   = (τ - 1) × (y - ŷ)   if y <  ŷ  (model overshot)

    Which can be written compactly as:
        L(y, ŷ, τ) = max(τ(y - ŷ),  (τ-1)(y - ŷ))

    Why this produces the correct quantile:
        Consider τ = 0.95 (the 95th percentile head).
        - If the model undershoots (true value is above prediction):
              penalty = 0.95 × error  — large penalty
        - If the model overshoots (true value is below prediction):
              penalty = 0.05 × error  — small penalty
        The model minimizes total loss by predicting high enough that
        it only undershoots 5% of the time — which is exactly the
        definition of the 95th percentile.

        For τ = 0.05 the logic inverts: the model learns to predict
        low enough that it only overshoots 5% of the time.

        For τ = 0.50 the penalties are symmetric (0.5 each way) —
        this is equivalent to minimizing mean absolute error, which
        converges to the median.

    Quantile crossing:
        Nothing in the loss prevents the 25th percentile head from
        predicting a higher value than the 75th percentile head for
        a given sample. This is called quantile crossing and is
        mathematically inconsistent. We handle it in two ways:
            1. The shared Transformer backbone creates implicit pressure
               toward consistent ordering — all heads see the same
               representation and tend to learn consistent mappings.
            2. At inference time we sort the quantile outputs to enforce
               monotonicity. We do NOT enforce this during training
               because adding ordering constraints to the loss makes
               optimization significantly harder and the violation rate
               is low in practice after convergence.

    Parameters
    ----------
    quantiles : list of float
        Target quantile levels in ascending order.
        Must match the order of columns in the model output tensor.
        E.g., [0.05, 0.25, 0.50, 0.75, 0.95]
    """

    def __init__(self, quantiles: List[float]):
        super().__init__()

        assert quantiles == sorted(quantiles), \
            "Quantiles must be in ascending order"
        assert all(0 < q < 1 for q in quantiles), \
            "All quantiles must be strictly between 0 and 1"

        self.quantiles = quantiles

        # Register quantile tensor as a buffer so it moves to the correct
        # device automatically when model.to(device) is called
        self.register_buffer(
            "quantile_tensor",
            torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(
        self,
        predictions : torch.Tensor,
        targets     : torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean pinball loss across all quantiles and all samples.

        Args:
            predictions : (batch, n_quantiles)
                          Model output — one predicted value per quantile.
            targets     : (batch, 1)
                          True normalized glucose values.

        Returns:
            Scalar loss — mean pinball loss averaged over the batch
            and over all quantiles.
        """
        # Expand targets to match prediction shape
        # (batch, 1) → (batch, n_quantiles)
        # Each target is compared against every quantile prediction
        targets_expanded = targets.expand_as(predictions)

        # Signed errors — positive means model undershot (true > predicted)
        #                  negative means model overshot  (true < predicted)
        errors = targets_expanded - predictions
        # Shape: (batch, n_quantiles)

        # Apply asymmetric penalty
        # For each quantile τ:
        #   undershoot (error > 0): penalty = τ × error
        #   overshoot  (error < 0): penalty = (τ-1) × error = (1-τ) × |error|
        #
        # torch.where selects element-wise:
        #   where error >= 0: use τ × error
        #   where error <  0: use (τ-1) × error
        #
        # self.quantile_tensor shape: (n_quantiles,)
        # Broadcasting handles the batch dimension automatically
        loss = torch.where(
            errors >= 0,
            self.quantile_tensor * errors,
            (self.quantile_tensor - 1.0) * errors,
        )
        # Shape: (batch, n_quantiles) — all values non-negative

        # Average over both batch and quantile dimensions
        return loss.mean()

    def per_quantile_loss(
        self,
        predictions : torch.Tensor,
        targets     : torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss broken down per quantile rather than averaged.

        Useful for monitoring training — if one quantile's loss diverges
        from the others it may indicate a problem with that head.

        Returns:
            (n_quantiles,) tensor — mean pinball loss per quantile
        """
        targets_expanded = targets.expand_as(predictions)
        errors = targets_expanded - predictions

        loss = torch.where(
            errors >= 0,
            self.quantile_tensor * errors,
            (self.quantile_tensor - 1.0) * errors,
        )
        # Average over batch only, keep quantile dimension
        return loss.mean(dim=0)

    def coverage(
        self,
        predictions : torch.Tensor,
        targets     : torch.Tensor,
    ) -> dict:
        """
        Compute empirical coverage for each symmetric interval defined
        by the quantile pairs.

        For quantiles [0.05, 0.25, 0.50, 0.75, 0.95]:
            90% interval: (q0.05, q0.95) — target coverage 0.90
            50% interval: (q0.25, q0.75) — target coverage 0.50

        This is a diagnostic tool — call it on validation data to monitor
        whether the model is learning well-calibrated intervals. If the
        90% interval only achieves 70% empirical coverage, the model is
        underestimating uncertainty.

        Returns:
            dict mapping interval name to empirical coverage float
        """
        targets_flat = targets.squeeze(1)
        n_q          = len(self.quantiles)
        results      = {}

        for i in range(n_q // 2):
            lower_idx = i
            upper_idx = n_q - 1 - i
            lower_tau = self.quantiles[lower_idx]
            upper_tau = self.quantiles[upper_idx]

            lower_preds = predictions[:, lower_idx]
            upper_preds = predictions[:, upper_idx]

            in_interval = (
                (targets_flat >= lower_preds) &
                (targets_flat <= upper_preds)
            )
            empirical_coverage = in_interval.float().mean().item()
            target_coverage    = upper_tau - lower_tau

            label = f"{round(target_coverage * 100)}%"
            results[label] = {
                "target"   : target_coverage,
                "empirical": empirical_coverage,
                "gap"      : empirical_coverage - target_coverage,
            }

        return results