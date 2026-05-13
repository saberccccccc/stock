# model.py - UltimateV7 截面Alpha模型
# V7改进：特征分组时序编码 + 跨股票Transformer + 可选GAT分支 + 多周期头
# V9改进：头部分支Dropout + 更强正则化
# GAT实现 (2026-05-12): 行业图注意力 + FusionGate融合
import zlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FeatureGrouper(nn.Module):
    """按聚合类型分组的特征编码器：last/sma5/sma20/vol5/vol20 -> 时序感知embedding"""

    def __init__(self, input_dim, hidden_dim, base_feat_dim, n_aggs=5):
        super().__init__()
        self.base_feat_dim = base_feat_dim
        self.n_aggs = n_aggs
        self.agg_feat_dim = base_feat_dim * n_aggs

        self.num_agg_heads = 2
        preferred_dim = 48
        max_per_agg = (hidden_dim - 1) // n_aggs
        if max_per_agg < self.num_agg_heads:
            raise ValueError(
                f"hidden_dim={hidden_dim} too small for n_aggs={n_aggs}; "
                f"need at least {n_aggs * self.num_agg_heads + 1}"
            )
        if preferred_dim * n_aggs < hidden_dim:
            self.per_agg_dim = preferred_dim
        else:
            self.per_agg_dim = (max_per_agg // self.num_agg_heads) * self.num_agg_heads
        agg_total = self.per_agg_dim * n_aggs
        self.extra_dim = hidden_dim - agg_total

        # 组内投影：每个聚合类型独立投影
        self.agg_proj = nn.ModuleList([
            nn.Linear(base_feat_dim, self.per_agg_dim)
            for _ in range(n_aggs)
        ])
        # 跨聚合类型特征交互（学习不同时间尺度的关系）
        self.cross_agg_attn = nn.MultiheadAttention(
            embed_dim=self.per_agg_dim, num_heads=self.num_agg_heads,
            batch_first=True, dropout=0.3
        )
        # 额外的skip-projection用于非聚合特征
        self.extra_proj = nn.Linear(
            input_dim - self.agg_feat_dim, self.extra_dim
        )

    def forward(self, x):
        # x: (B, N, F_total)
        B, N, F = x.shape
        agg_feat = x[..., :self.agg_feat_dim]
        extra_feat = x[..., self.agg_feat_dim:]

        # 每个聚合类型独立编码
        agg_parts = []
        for i in range(self.n_aggs):
            chunk = agg_feat[..., i * self.base_feat_dim:(i + 1) * self.base_feat_dim]
            proj = self.agg_proj[i](chunk)  # (B, N, h/n_aggs)
            agg_parts.append(proj)

        # 跨聚合类型注意力：把 n_aggs 当作"序列"维度
        # (B*N, n_aggs, h/n_aggs)
        stacked = torch.stack(agg_parts, dim=1).reshape(B * N, self.n_aggs, -1)
        attn_out, _ = self.cross_agg_attn(stacked, stacked, stacked)
        attn_out = attn_out.reshape(B, N, self.n_aggs, -1)
        agg_out = attn_out.reshape(B, N, -1)  # (B, N, hidden_dim')

        # 额外特征（rank + industry_relative）直接投影并拼接
        extra_out = self.extra_proj(extra_feat)
        return torch.cat([agg_out, extra_out], dim=-1)  # (B, N, hidden_dim)


class FusionGate(nn.Module):
    """自适应融合 Transformer 和 GAT 分支"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, trans_out, gat_out):
        # trans_out, gat_out: (B, N, hidden_dim)
        gate_input = torch.cat([trans_out, gat_out], dim=-1)
        gate = self.gate(gate_input)  # (B, N, 1)
        return gate * trans_out + (1 - gate) * gat_out


class UltimateV7Model(nn.Module):
    """
    V7改进：
    - FeatureGrouper: 对5种聚合（last/sma5/sma20/vol5/vol20）做组内+跨组编码
    - Transformer: 跨股票注意力学习排序关系
    - GAT分支: 行业图注意力（可选）
    - FusionGate: 自适应融合两支
    - 多周期预测头: h1/h3/h5/h7 四个horizon
    """

    def __init__(self, input_dim, base_feat_dim, n_aggs=5,
                 hidden_dim=256, n_heads=8, n_layers=4,
                 n_horizons=4, n_alpha=4, use_gat=False, regime_dim=87,
                 num_industries=83, industry_emb_dim=16):
        super().__init__()
        self.use_gat = use_gat
        self.hidden_dim = hidden_dim
        self.n_horizons = n_horizons
        self.industry_emb_dim = industry_emb_dim
        self.num_industries = num_industries

        # 特征分组编码
        self.feature_grouper = FeatureGrouper(
            input_dim, hidden_dim, base_feat_dim, n_aggs
        )

        # 行业embedding（真实行业 + 1个未知行业）
        self.industry_embed = nn.Embedding(num_industries + 1, industry_emb_dim)
        self.industry_proj = nn.Linear(hidden_dim + industry_emb_dim, hidden_dim)

        # 排名嵌入
        self.rank_embed = nn.Embedding(512, hidden_dim)

        # Transformer 编码器（跨股票注意力）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.35, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 多头Alpha
        self.alpha_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_alpha)
        ])
        self.alpha_gate = nn.Sequential(
            nn.Linear(hidden_dim, n_alpha), nn.Softmax(dim=-1)
        )

        # 多周期预测头
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_horizons)
        ])

        # 市场状态编码(regime_dim = 股票级风险 + 市场特征 + 可选宏观特征)
        self.regime_proj = nn.Linear(regime_dim, hidden_dim)

        # GAT分支：1层GATConv，行业全连接图
        if use_gat:
            self.gat_conv = GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.3)
            self.gat_proj = nn.Linear(hidden_dim, hidden_dim)
            self.gat_norm = nn.LayerNorm(hidden_dim)
            self.fusion_gate = FusionGate(hidden_dim)
        self._eval_edge_cache = OrderedDict()
        self._eval_edge_cache_max_size = 32

    def _build_rank_embed(self, X, mask):
        """基于第一维特征构建排名嵌入"""
        B, N, _ = X.shape
        # 在有效股票内排名
        ranks = torch.zeros(B, N, dtype=torch.long, device=X.device)
        for b in range(B):
            valid_idx = mask[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) > 1:
                vals = X[b, valid_idx, 0]
                sorted_idx = torch.argsort(torch.argsort(vals))
                clamped = torch.clamp(sorted_idx, 0, 511)
                ranks[b, valid_idx] = clamped
        return self.rank_embed(ranks)  # (B, N, hidden_dim)

    def build_industry_edges(self, industry_ids, mask, max_edges_per_stock=5):
        """Build same-industry graph edges on CPU."""
        device = industry_ids.device
        ids_cpu = industry_ids.cpu()
        mask_cpu = mask.cpu()

        if not self.training:
            ids_arr = ids_cpu.numpy()
            mask_arr = mask_cpu.numpy()
            ids_crc = zlib.crc32(ids_arr.tobytes())
            mask_crc = zlib.crc32(mask_arr.tobytes())
            cache_key = (ids_arr.shape, ids_crc, mask_crc, max_edges_per_stock)
            if cache_key in self._eval_edge_cache:
                cached_edges = self._eval_edge_cache.pop(cache_key)
                self._eval_edge_cache[cache_key] = cached_edges
                return [e.to(device) for e in cached_edges]

        B, N = ids_cpu.shape
        batch_edges = []
        for b in range(B):
            valid = mask_cpu[b]
            ids = ids_cpu[b]
            unique_ids = ids[(ids >= 0) & valid].unique()
            edges_list = []
            for uid in unique_ids:
                idx_global = ((ids == uid) & valid).nonzero(as_tuple=True)[0]
                n_ind = len(idx_global)
                if n_ind < 2:
                    continue
                k = min(max_edges_per_stock, n_ind - 1)
                src_parts = []
                dst_parts = []
                for src_pos in range(n_ind):
                    if self.training:
                        perm = torch.randperm(n_ind - 1)[:k]
                        candidates = torch.cat([idx_global[:src_pos], idx_global[src_pos + 1:]])
                        dst_nodes = candidates[perm]
                    else:
                        offsets = torch.arange(1, k + 1)
                        dst_nodes = idx_global[(src_pos + offsets) % n_ind]
                    src_parts.append(idx_global[src_pos].repeat(k))
                    dst_parts.append(dst_nodes)
                src_global = torch.cat(src_parts)
                dst_global = torch.cat(dst_parts)
                edges_list.append(torch.stack([src_global, dst_global]))
            if edges_list:
                e = torch.cat(edges_list, dim=1)
                e = torch.cat([e, e.flip(0)], dim=1)
            else:
                e = torch.zeros(2, 0, dtype=torch.long)
            batch_edges.append(e)

        if not self.training:
            self._eval_edge_cache[cache_key] = batch_edges
            while len(self._eval_edge_cache) > self._eval_edge_cache_max_size:
                self._eval_edge_cache.popitem(last=False)
        return [e.to(device) for e in batch_edges]

    def _gat_forward(self, h, mask, industry_ids, trans_out=None):
        """GAT forward: single GATConv layer, industry subgraph propagation"""
        B, N, H = h.shape
        fallback = trans_out if trans_out is not None else h
        gat_out = fallback.clone()
        for b in range(B):
            valid = mask[b]
            n_valid = valid.sum().item()
            if n_valid < 5:
                continue
            batch_edges = self.build_industry_edges(
                industry_ids[b:b+1], mask[b:b+1]
            )
            edges = batch_edges[0]
            if edges.shape[1] == 0:
                del batch_edges, edges
                continue
            x = self.gat_proj(h[b])
            x = self.gat_conv(x, edges)
            x = F.gelu(x)
            x = self.gat_norm(x)
            gat_out[b] = x
            del batch_edges, edges, x
        return gat_out

    def forward(self, X, risk_cont, mask, industry_ids):
        """
        Args:
            X: (B, N, F) features
            mask: (B, N) bool, True=valid
            industry_ids: (B, N) long, industry ID
        Returns:
            alpha_raw: (B, N) alphas
            horizon_preds: (B, N, n_horizons) preds
        """


        B, N, _ = X.shape

        # 1. 特征分组编码
        h = self.feature_grouper(X)  # (B, N, hidden_dim)

        # 2. 行业embedding拼接并投影
        # 将-1映射到num_industries（作为未知行业）
        industry_ids_valid = torch.where(industry_ids >= 0, industry_ids, self.num_industries)
        industry_ids_valid = torch.clamp(industry_ids_valid, 0, self.num_industries)
        industry_emb = self.industry_embed(industry_ids_valid)  # (B, N, industry_emb_dim)
        h = torch.cat([h, industry_emb], dim=-1)  # (B, N, hidden_dim + industry_emb_dim)
        h = self.industry_proj(h)  # (B, N, hidden_dim)

        # 3. 排名嵌入
        rank_emb = self._build_rank_embed(X, mask)
        h = h + rank_emb

        # 4. 市场状态
        mask_f = mask.float().unsqueeze(-1)
        regime_sum = (risk_cont * mask_f).sum(dim=1)
        regime_cnt = mask_f.sum(dim=1).clamp(min=1)
        regime_avg = regime_sum / regime_cnt
        regime_h = self.regime_proj(regime_avg).unsqueeze(1)

        # 4. Transformer（跨股票注意力）
        trans_out = self.transformer(h, src_key_padding_mask=~mask)
        trans_out = trans_out + regime_h

        # 5. GAT分支：行业图注意力传播
        if self.use_gat:
            gat_out = self._gat_forward(h, mask, industry_ids, trans_out=trans_out)
            h_out = self.fusion_gate(trans_out, gat_out)
        else:
            h_out = trans_out

        # 6. Alpha 头
        alphas = torch.cat([head(h_out) for head in self.alpha_heads], dim=-1)
        gate = self.alpha_gate(h_out)
        alpha_raw = (alphas * gate).sum(dim=-1)

        # 7. 多周期预测头
        horizon_preds = torch.cat(
            [head(h_out) for head in self.horizon_heads], dim=-1
        )  # (B, N, n_horizons)

        return alpha_raw, alphas, horizon_preds
