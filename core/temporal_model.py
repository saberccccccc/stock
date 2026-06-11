"""Temporal + cross-section alpha model.

The model keeps the old output contract:
    alpha_raw, alphas, horizon_preds

It only changes the input contract by adding X_seq, a true per-stock sequence:
    forward(X, X_seq, risk_cont, mask, industry_ids)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x):
        y = self.conv(x)
        if self.left_padding > 0:
            y = y[..., :-self.left_padding]
        return y


class FeatureGrouper(nn.Module):
    """Local copy of the V7 feature grouper without torch_geometric dependency."""

    def __init__(self, input_dim, hidden_dim, agg_groups):
        super().__init__()
        self.agg_groups = agg_groups
        self.group_dims = [base * n for base, n, _ in agg_groups]
        self.total_agg_dim = sum(self.group_dims)
        self.group_n_aggs = [n for _, n, _ in agg_groups]

        total_slots = sum(self.group_n_aggs)
        num_heads = 2
        preferred_dim = 32
        if preferred_dim * total_slots < hidden_dim:
            self.per_slot_dim = preferred_dim
        else:
            max_per_slot = max(2, (hidden_dim - 1) // max(total_slots, 1))
            self.per_slot_dim = max(2, (max_per_slot // num_heads) * num_heads)

        agg_total = self.per_slot_dim * total_slots
        self.extra_dim = hidden_dim - agg_total
        if self.extra_dim <= 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} too small for agg_groups={agg_groups}; "
                f"need more than {agg_total}"
            )

        self.agg_projs = nn.ModuleList()
        self.cross_agg_attns = nn.ModuleList()
        self.group_residuals = nn.ModuleList()
        for base_dim, n_aggs, dropout in agg_groups:
            projs = nn.ModuleList([nn.Linear(base_dim, self.per_slot_dim) for _ in range(n_aggs)])
            self.agg_projs.append(projs)
            if n_aggs > 1:
                attn = nn.MultiheadAttention(self.per_slot_dim, num_heads, batch_first=True, dropout=dropout)
            else:
                attn = None
            self.cross_agg_attns.append(attn)
            self.group_residuals.append(nn.Linear(base_dim * n_aggs, n_aggs * self.per_slot_dim))

        extra_in = input_dim - self.total_agg_dim
        if extra_in <= 0:
            raise ValueError(f"input_dim={input_dim} must exceed aggregated dim={self.total_agg_dim}")
        self.extra_proj = nn.Linear(extra_in, self.extra_dim)

    def forward(self, x):
        batch_size, n_stocks, _ = x.shape
        group_outputs = []
        offset = 0
        for g_idx, (base_dim, n_aggs, _) in enumerate(self.agg_groups):
            group_dim = base_dim * n_aggs
            group_feat = x[..., offset:offset + group_dim]
            offset += group_dim

            parts = []
            for i in range(n_aggs):
                chunk = group_feat[..., i * base_dim:(i + 1) * base_dim]
                parts.append(self.agg_projs[g_idx][i](chunk))

            attn = self.cross_agg_attns[g_idx]
            if attn is not None:
                stacked = torch.stack(parts, dim=1).reshape(batch_size * n_stocks, n_aggs, -1)
                attn_out, _ = attn(stacked, stacked, stacked)
                group_out = attn_out.reshape(batch_size, n_stocks, -1)
            else:
                group_out = parts[0]

            group_out = group_out + self.group_residuals[g_idx](group_feat)
            group_outputs.append(group_out)

        agg_out = torch.cat(group_outputs, dim=-1)
        extra_out = self.extra_proj(x[..., self.total_agg_dim:])
        return torch.cat([agg_out, extra_out], dim=-1)


class TemporalConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.dropout(F.gelu(x))
        return residual + x


class StockTemporalEncoder(nn.Module):
    """Encode one stock's recent trajectory into a hidden vector."""

    def __init__(self, seq_dim, hidden_dim=256, temporal_dim=128, n_blocks=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(seq_dim, temporal_dim)
        dilations = [1, 2, 4, 8][:n_blocks]
        self.blocks = nn.ModuleList([
            TemporalConvBlock(temporal_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        self.out = nn.Sequential(
            nn.Linear(temporal_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x_seq):
        batch_size, n_stocks, lookback, seq_dim = x_seq.shape
        x = x_seq.reshape(batch_size * n_stocks, lookback, seq_dim)
        x = self.input_proj(x).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        last = x[..., -1]
        mean = x.mean(dim=-1)
        h = self.out(torch.cat([last, mean], dim=-1))
        return h.reshape(batch_size, n_stocks, -1)


class TemporalFusionGate(nn.Module):
    def __init__(self, hidden_dim, gate_init=0.0, mode="blend_norm"):
        super().__init__()
        self.mode = str(mode)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate[2].bias, float(gate_init))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cross_h, temporal_h):
        gate = self.gate(torch.cat([cross_h, temporal_h, cross_h - temporal_h], dim=-1))
        if self.mode == "residual_add":
            return cross_h + gate * temporal_h
        fused = gate * cross_h + (1.0 - gate) * temporal_h
        return self.norm(fused + cross_h)


class TemporalResidualAdapter(nn.Module):
    """Small gated residual path for re-injecting temporal information."""

    def __init__(self, hidden_dim, dropout=0.10, gate_init=-3.0):
        super().__init__()
        self.temporal_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.constant_(self.gate.bias, gate_init)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, temporal_h):
        gate = torch.sigmoid(self.gate(torch.cat([h, temporal_h], dim=-1)))
        return self.norm(h + gate * self.temporal_proj(temporal_h))


class TemporalCrossAlphaModel(nn.Module):
    """A conservative temporal extension of UltimateV7Model.

    Design:
    - current cross-section features still go through FeatureGrouper;
    - per-stock sequences go through a causal TCN;
    - a learned gate fuses the two views before cross-stock Transformer layers;
    - heads and output shapes match the existing training losses.
    """

    def __init__(
        self,
        input_dim,
        seq_dim,
        agg_groups=None,
        hidden_dim=256,
        temporal_dim=128,
        n_temporal_blocks=3,
        n_heads=8,
        n_layers=2,
        n_horizons=4,
        n_alpha=4,
        regime_dim=56,
        num_industries=83,
        industry_emb_dim=16,
        dropout=0.35,
        temporal_dropout=0.10,
        temporal_fusion_cross_gate_init=0.0,
        temporal_fusion_mode="blend_norm",
        transformer_norm_first=True,
        temporal_residual_adapters=False,
        temporal_adapter_dropout=0.10,
        temporal_adapter_gate_init=-3.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.seq_dim = seq_dim
        self.n_horizons = n_horizons
        self.num_industries = num_industries
        self.temporal_residual_adapters = bool(temporal_residual_adapters)
        self.temporal_fusion_cross_gate_init = float(temporal_fusion_cross_gate_init)
        self.temporal_fusion_mode = str(temporal_fusion_mode)
        self.transformer_norm_first = bool(transformer_norm_first)
        self.temporal_adapter_gate_init = float(temporal_adapter_gate_init)

        if agg_groups is None:
            agg_groups = [(23, 5, 0.1), (7, 2, 0.0)]
        self.agg_groups = agg_groups

        self.feature_grouper = FeatureGrouper(input_dim, hidden_dim, agg_groups)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.trans_input_proj = nn.Linear(input_dim, hidden_dim)

        self.temporal_encoder = StockTemporalEncoder(
            seq_dim=seq_dim,
            hidden_dim=hidden_dim,
            temporal_dim=temporal_dim,
            n_blocks=n_temporal_blocks,
            dropout=temporal_dropout,
        )
        self.temporal_fusion = TemporalFusionGate(
            hidden_dim,
            gate_init=temporal_fusion_cross_gate_init,
            mode=temporal_fusion_mode,
        )
        if self.temporal_residual_adapters:
            self.temporal_adapter_pre = TemporalResidualAdapter(
                hidden_dim, dropout=temporal_adapter_dropout, gate_init=temporal_adapter_gate_init
            )
            self.temporal_adapter_mid = TemporalResidualAdapter(
                hidden_dim, dropout=temporal_adapter_dropout, gate_init=temporal_adapter_gate_init
            )
            self.temporal_adapter_out = TemporalResidualAdapter(
                hidden_dim, dropout=temporal_adapter_dropout, gate_init=temporal_adapter_gate_init
            )
        else:
            self.temporal_adapter_pre = nn.Identity()
            self.temporal_adapter_mid = nn.Identity()
            self.temporal_adapter_out = nn.Identity()

        self.industry_embed = nn.Embedding(num_industries + 1, industry_emb_dim)
        self.industry_proj = nn.Linear(hidden_dim + industry_emb_dim, hidden_dim)
        self.rank_embed = nn.Embedding(512, hidden_dim)
        self.regime_proj = nn.Linear(regime_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=self.transformer_norm_first,
        )
        n_per_block = max(1, n_layers // 2)
        self.trans1 = nn.TransformerEncoder(enc_layer, num_layers=n_per_block)
        self.trans2 = nn.TransformerEncoder(enc_layer, num_layers=n_per_block)

        self.alpha_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_alpha)
        ])
        self.alpha_gate = nn.Sequential(nn.Linear(hidden_dim, n_alpha), nn.Softmax(dim=-1))

        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_horizons)
        ])

    def _build_rank_embed(self, X, mask):
        batch_size, n_stocks, _ = X.shape
        ranks = torch.zeros(batch_size, n_stocks, dtype=torch.long, device=X.device)
        for b in range(batch_size):
            valid_idx = mask[b].nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 1:
                vals = X[b, valid_idx, 0]
                sorted_idx = torch.argsort(torch.argsort(vals))
                ranks[b, valid_idx] = torch.clamp(sorted_idx, 0, 511)
        return self.rank_embed(ranks)

    def forward(self, X, X_seq, risk_cont, mask, industry_ids):
        cross_h = self.feature_grouper(X) + self.input_proj(X)
        temporal_h = self.temporal_encoder(X_seq)
        h = self.temporal_fusion(cross_h, temporal_h)
        if self.temporal_residual_adapters:
            h = self.temporal_adapter_pre(h, temporal_h)

        industry_ids_valid = torch.where(industry_ids >= 0, industry_ids, self.num_industries)
        industry_ids_valid = torch.clamp(industry_ids_valid, 0, self.num_industries)
        industry_emb = self.industry_embed(industry_ids_valid)
        h = h + self.industry_proj(torch.cat([h, industry_emb], dim=-1))
        h = h + self._build_rank_embed(X, mask)
        h = h + self.trans_input_proj(X)

        mask_f = mask.float().unsqueeze(-1)
        regime_sum = (risk_cont * mask_f).sum(dim=1)
        regime_count = mask_f.sum(dim=1).clamp(min=1.0)
        regime_h = self.regime_proj(regime_sum / regime_count).unsqueeze(1)

        t1 = self.trans1(h, src_key_padding_mask=~mask) + regime_h
        if self.temporal_residual_adapters:
            t1 = self.temporal_adapter_mid(t1, temporal_h)
        h_out = self.trans2(t1, src_key_padding_mask=~mask) + regime_h
        if self.temporal_residual_adapters:
            h_out = self.temporal_adapter_out(h_out, temporal_h)

        alphas = torch.cat([head(h_out) for head in self.alpha_heads], dim=-1)
        gate = self.alpha_gate(h_out)
        alpha_raw = (alphas * gate).sum(dim=-1)
        horizon_preds = torch.cat([head(h_out) for head in self.horizon_heads], dim=-1)
        return alpha_raw, alphas, horizon_preds

    def arch_config(self):
        return {
            "model": "TemporalCrossAlphaModel",
            "hidden_dim": self.hidden_dim,
            "temporal_dim": self.temporal_dim,
            "seq_dim": self.seq_dim,
            "n_horizons": self.n_horizons,
            "num_industries": self.num_industries,
            "agg_groups": self.agg_groups,
            "temporal_residual_adapters": self.temporal_residual_adapters,
            "temporal_fusion_cross_gate_init": self.temporal_fusion_cross_gate_init,
            "temporal_fusion_mode": self.temporal_fusion_mode,
            "transformer_norm_first": self.transformer_norm_first,
            "temporal_adapter_gate_init": self.temporal_adapter_gate_init,
        }
