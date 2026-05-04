"""
model.py
========
Quantile Regression Transformer for blood glucose prediction.

Architecture:
    Input (batch, INPUT_WINDOW, N_FEATURES)
        → Feature projection to d_model
        → Positional encoding
        → N × TransformerEncoderLayer
        → Sequence pooling (attention-weighted mean)
        → Q × independent quantile heads

Key design decisions:
    1. Attention-weighted pooling rather than mean pooling — the model
       learns which timesteps carry the most predictive information rather
       than treating all positions equally.
    2. Separate quantile heads with a hidden layer each — allows each
       quantile to learn different non-linear mappings from the shared
       representation, which matters for the tail quantiles (0.05, 0.95)
       whose relationship to the input differs from the median.
    3. Mask convention flip — Dataset uses True=valid, PyTorch Transformer
       uses True=ignore. The flip is explicit in forward().
    4. No output activation — quantile predictions are unbounded scalars
       in normalized glucose space. Clamping to physiological bounds
       happens at evaluation time, not during training.
"""

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from tokenizer import INPUT_WINDOW, N_FEATURES, PREDICTION_HORIZON


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TransformerConfig:
    """
    All model hyperparameters in one place.

    Sizing rationale:
        These defaults are intentionally modest — d_model=64 with 4 heads
        and 3 layers gives ~200K parameters, which is appropriate for a
        dataset of ~50K training sequences. Larger models (d_model=128+)
        risk overfitting on this dataset size and are harder to deploy
        on edge hardware later.

        Rule of thumb used here: roughly 4-5 training sequences per
        model parameter. With ~50K sequences and 200K params we're at
        the conservative end, which is correct for a first model.
    """
    # Architecture
    d_model        : int   = 64       # Hidden dimension — must be divisible by n_heads
    n_heads        : int   = 4        # Attention heads — each attends to d_model/n_heads=16 dims
    n_layers       : int   = 3        # Encoder layers
    d_feedforward  : int   = 128      # FFN hidden dim — typically 2-4× d_model
    dropout        : float = 0.1      # Applied after attention and FFN

    # Input/output dimensions — derived from tokenizer constants
    n_input_features : int = N_FEATURES    # 6
    seq_len          : int = INPUT_WINDOW  # 24

    # Quantile regression
    quantiles : List[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95]
    )

    # Pooling strategy
    # 'attention' — learned weighted average over sequence positions
    # 'mean'      — simple unweighted average (faster, slightly worse)
    # 'last'      — use only the final timestep (common but wastes context)
    pooling : str = 'attention'

    @property
    def n_quantiles(self) -> int:
        return len(self.quantiles)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"n_heads ({self.n_heads})"
        )
        assert self.pooling in ('attention', 'mean', 'last'), (
            f"pooling must be 'attention', 'mean', or 'last'"
        )
        assert all(0 < q < 1 for q in self.quantiles), (
            "All quantiles must be strictly between 0 and 1"
        )
        assert self.quantiles == sorted(self.quantiles), (
            "Quantiles must be in ascending order"
        )


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention Is All You Need'.

    Adds a fixed position-dependent signal to token embeddings so the
    Transformer can distinguish between timesteps. Without this, the
    self-attention mechanism is permutation-invariant — it would treat
    the sequence as an unordered set.

    We use sinusoidal (not learned) positional encoding because:
        1. Our sequences are always exactly INPUT_WINDOW=24 long —
           there is no variable-length generalization needed.
        2. Sinusoidal encoding explicitly encodes relative distances
           between positions via the dot product of their encodings,
           which is useful for glucose dynamics where "30 minutes ago"
           has different meaning than "5 minutes ago."
        3. Fewer parameters to learn on a small dataset.

    Note: time-of-day information is already in f2/f3 (sin/cos hour).
    This positional encoding captures *position within the input window*
    (i.e., "this is the 5th token") — a different and complementary signal.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the sinusoidal table once at construction time
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims

        # Shape: (1, max_len, d_model) — batch dim for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Attention Pooling ─────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learned weighted average over the sequence dimension.

    Rather than treating all 24 timesteps as equally informative when
    collapsing the sequence to a fixed-size vector, this learns a scalar
    attention weight for each position. The final representation is the
    attention-weighted sum of all token representations.

    The pooling attention is separate from the Transformer's self-attention
    — it operates on the Transformer's output and decides which timesteps'
    final representations matter most for the prediction.

    Practically: the model can learn "the last few timesteps before the
    prediction window are most informative" without us having to hard-code
    that assumption.

    Masked positions (gap tokens) are excluded from the weighted average
    by setting their logits to -inf before softmax, ensuring they
    contribute zero weight regardless of their representation values.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Single linear layer maps each token's representation to a scalar score
        self.attention = nn.Linear(d_model, 1)

    def forward(
        self,
        x            : torch.Tensor,
        padding_mask : torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x            : (batch, seq_len, d_model) — Transformer output
            padding_mask : (batch, seq_len) bool — True = VALID token
                           (Dataset convention — True means attend to this)

        Returns:
            pooled : (batch, d_model) — weighted average representation
        """
        # Compute raw attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Mask out invalid positions by setting their logits to -inf
        # After softmax, -inf → 0.0, so masked tokens contribute nothing
        # We use the Dataset convention here (True = valid), so we mask
        # where padding_mask is False
        scores = scores.masked_fill(~padding_mask, float('-inf'))

        # Normalize to get attention weights
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum over sequence positions
        # (batch, seq_len, 1) × (batch, seq_len, d_model) → (batch, d_model)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)

        return pooled


# ── Quantile Head ─────────────────────────────────────────────────────────────

class QuantileHead(nn.Module):
    """
    Single output head predicting one quantile of the glucose distribution.

    Each head is a small two-layer MLP applied to the pooled sequence
    representation. Using separate heads per quantile (rather than one
    shared head with multiple outputs) allows each quantile to learn a
    different non-linear mapping from the shared representation.

    This matters most for the tail quantiles (0.05, 0.95) — their
    relationship to the input sequence is different from the median
    because they need to capture the conditions under which extreme
    outcomes occur, not just average dynamics.

    Architecture per head:
        d_model → d_model//2 → GELU → Dropout → 1
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, d_model)
        Returns:
            (batch, 1) — predicted quantile value in normalized glucose space
        """
        return self.net(x)


# ── Main Model ────────────────────────────────────────────────────────────────

class GlucoseTransformer(nn.Module):
    """
    Quantile Regression Transformer for 30-minute blood glucose prediction.

    Forward pass:
        1. Project 6 input features to d_model dimensions
        2. Add sinusoidal positional encoding
        3. Pass through N Transformer encoder layers with attention masking
        4. Pool sequence to fixed-size vector (attention-weighted)
        5. Pass through Q independent quantile heads
        6. Return (batch, Q) tensor of quantile predictions

    The attention mask is flipped inside forward() from Dataset convention
    (True=valid) to PyTorch Transformer convention (True=ignore).
    This flip is documented at the point it happens.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # ── Input projection ──────────────────────────────────────────────────
        # Projects the 6-dimensional feature vector at each timestep into the
        # d_model-dimensional space that the Transformer operates in.
        # A single linear layer is sufficient here — the Transformer layers
        # will learn complex feature interactions; the projection just puts
        # features into the right dimensional space.
        self.input_projection = nn.Linear(
            config.n_input_features, config.d_model
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            d_model  = config.d_model,
            max_len  = config.seq_len + 10,  # +10 for safety margin
            dropout  = config.dropout,
        )

        # ── Transformer encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = config.d_model,
            nhead           = config.n_heads,
            dim_feedforward = config.d_feedforward,
            dropout         = config.dropout,
            activation      = 'gelu',
            batch_first     = True,   # (batch, seq, features) convention
            norm_first      = True,   # Pre-norm (more stable than post-norm
                                      # for smaller models and datasets)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = config.n_layers,
            # enable_nested_tensor=False avoids a PyTorch warning when
            # using src_key_padding_mask with batch_first=True
            enable_nested_tensor = False,
        )

        # ── Sequence pooling ──────────────────────────────────────────────────
        if config.pooling == 'attention':
            self.pooling_layer = AttentionPooling(config.d_model)
        # mean and last are handled directly in forward()

        # ── Quantile heads ────────────────────────────────────────────────────
        self.quantile_heads = nn.ModuleList([
            QuantileHead(config.d_model, config.dropout)
            for _ in config.quantiles
        ])

        # ── Weight initialization ─────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for stable training.

        Linear layers: Xavier uniform (standard for Transformer FFN layers).
        Biases: zero initialization.
        The Transformer's internal weights are already initialized by PyTorch.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        features       : torch.Tensor,
        attention_mask : torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features       : (batch, seq_len, n_features)
                             Input token sequence from DataLoader.
            attention_mask : (batch, seq_len) bool
                             Dataset convention: True = valid token.

        Returns:
            quantile_preds : (batch, n_quantiles)
                             Predicted quantile values in normalized
                             glucose space. Columns correspond to
                             config.quantiles in order.
        """
        # ── 1. Project features to model dimension ────────────────────────────
        x = self.input_projection(features)
        # Shape: (batch, seq_len, d_model)

        # ── 2. Add positional encoding ────────────────────────────────────────
        x = self.pos_encoding(x)
        # Shape: (batch, seq_len, d_model)

        # ── 3. Transformer encoder ────────────────────────────────────────────
        # MASK CONVENTION FLIP:
        # Dataset:            True  = valid token (attend to it)
        # PyTorch Transformer: True  = ignore token (do NOT attend to it)
        # We flip here, at the boundary between our code and PyTorch's.
        padding_mask_transformer = ~attention_mask
        # Shape: (batch, seq_len) — True now means "ignore this position"

        x = self.transformer(
            src                = x,
            src_key_padding_mask = padding_mask_transformer,
        )
        # Shape: (batch, seq_len, d_model)

        # ── 4. Pool sequence to fixed-size vector ─────────────────────────────
        if self.config.pooling == 'attention':
            # Pass Dataset-convention mask (True=valid) to AttentionPooling
            # which handles the masking internally
            pooled = self.pooling_layer(x, attention_mask)

        elif self.config.pooling == 'mean':
            # Masked mean — exclude invalid tokens from the average
            mask_float = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)

        else:  # 'last'
            # Use only the final valid token in each sequence
            # Find the last valid position per sample
            last_valid = attention_mask.long().cumsum(dim=1).argmax(dim=1)
            pooled = x[torch.arange(x.size(0)), last_valid]

        # Shape: (batch, d_model)

        # ── 5. Quantile heads ─────────────────────────────────────────────────
        quantile_preds = torch.cat(
            [head(pooled) for head in self.quantile_heads],
            dim=1
        )
        # Shape: (batch, n_quantiles)

        return quantile_preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> Dict[str, int]:
        """Returns parameter count broken down by component."""
        components = {
            'input_projection' : self.input_projection,
            'pos_encoding'     : self.pos_encoding,
            'transformer'      : self.transformer,
            'quantile_heads'   : self.quantile_heads,
        }
        if self.config.pooling == 'attention':
            components['pooling'] = self.pooling_layer

        return {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in components.items()
        }