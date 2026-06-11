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
    """Multi-group 特征分组编码器。

    每个 agg group 独立做 per-aggregation 投影 + 组内 cross-agg attention，
    支持不同数量的聚合方式（高频 5种、低频 2种）。

    Args:
        input_dim: 输入总维度 X
        hidden_dim: 输出总维度
        agg_groups: list of (base_feat_dim, n_aggs, dropout)
            e.g. [(23, 5, 0.1), (7, 2, 0.0)]
            各组输出维度按 slots 数均匀分配
    """

    def __init__(self, input_dim, hidden_dim, agg_groups):
        super().__init__()
        self.agg_groups = agg_groups
        self.group_dims = [base * n for base, n, _ in agg_groups]
        self.total_agg_dim = sum(self.group_dims)
        self.group_n_aggs = [n for _, n, _ in agg_groups]

        total_slots = sum(self.group_n_aggs)
        num_heads = 2
        preferred_dim = 32  # 高频5+低频2=7 slots, 32×7=224 → extra=32
        if preferred_dim * total_slots < hidden_dim:
            self.per_slot_dim = preferred_dim
        else:
            max_per_slot = (hidden_dim - 1) // total_slots
            self.per_slot_dim = (max_per_slot // num_heads) * num_heads

        agg_total = self.per_slot_dim * total_slots
        self.extra_dim = hidden_dim - agg_total

        # Build per-group modules
        self.agg_projs = nn.ModuleList()
        self.cross_agg_attns = nn.ModuleList()
        self.group_residuals = nn.ModuleList()  # 组内残差投影
        for base_dim, n_aggs, dropout in agg_groups:
            projs = nn.ModuleList([nn.Linear(base_dim, self.per_slot_dim) for _ in range(n_aggs)])
            self.agg_projs.append(projs)
            if n_aggs > 1:
                attn = nn.MultiheadAttention(self.per_slot_dim, num_heads, batch_first=True, dropout=dropout)
            else:
                attn = None
            self.cross_agg_attns.append(attn)
            # 残差投影：原始组特征 → 组输出维度
            self.group_residuals.append(nn.Linear(base_dim * n_aggs, n_aggs * self.per_slot_dim))

        self.extra_proj = nn.Linear(input_dim - self.total_agg_dim, self.extra_dim)

    def forward(self, x):
        B, N, _ = x.shape

        group_outputs = []
        offset = 0
        for g_idx in range(len(self.agg_groups)):
            base_dim, n_aggs, _ = self.agg_groups[g_idx]
            g_dim = self.group_dims[g_idx]
            group_feat = x[..., offset:offset + g_dim]
            offset += g_dim

            parts = []
            for i in range(n_aggs):
                chunk = group_feat[..., i * base_dim:(i + 1) * base_dim]
                parts.append(self.agg_projs[g_idx][i](chunk))

            attn = self.cross_agg_attns[g_idx]
            if attn is not None:
                stacked = torch.stack(parts, dim=1).reshape(B * N, n_aggs, -1)
                attn_out, _ = attn(stacked, stacked, stacked)
                group_out = attn_out.reshape(B, N, -1)
            else:
                group_out = parts[0]

            # 组内残差：原始特征投影后加到 attention 输出
            group_out = group_out + self.group_residuals[g_idx](group_feat)
            group_outputs.append(group_out)

        agg_out = torch.cat(group_outputs, dim=-1)
        extra_feat = x[..., self.total_agg_dim:]
        extra_out = self.extra_proj(extra_feat)
        return torch.cat([agg_out, extra_out], dim=-1)


class FusionGate(nn.Module):
    """自适应融合 Transformer 和 GAT 分支"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, trans_out, gat_out):
        # 逐维度 gate：每个特征维独立决定信 Transformer 还是信 GAT
        gate_input = torch.cat([trans_out, gat_out], dim=-1)
        gate = self.gate(gate_input)  # (B, N, H)
        return gate * trans_out + (1 - gate) * gat_out


class CrossIndustryAttention(nn.Module):
    """GAT输出 → 按行业pool → 跨行业attention → 逐股票门控吸收"""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ind_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.ind_norm = nn.LayerNorm(hidden_dim)
        # 逐股票门控：决定吸收多少跨行业信号
        self.stock_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, industry_ids, mask):
        B, N, H = x.shape
        output = x.clone()
        for b in range(B):
            valid = mask[b]
            ids = industry_ids[b][valid]
            feats = x[b][valid]
            unique_ids = ids[ids >= 0].unique()
            if len(unique_ids) < 3:
                continue

            nodes, idx_map = [], {}
            for uid in unique_ids:
                idx_map[uid.item()] = len(nodes)
                nodes.append(feats[ids == uid].mean(0))
            nodes = torch.stack(nodes)

            updated = self.ind_attn(nodes.unsqueeze(0), nodes.unsqueeze(0), nodes.unsqueeze(0))[0]
            updated = self.ind_norm(updated.squeeze(0) + nodes)  # (n_inds, H)

            valid_pos = valid.nonzero(as_tuple=True)[0]
            for uid, pos in idx_map.items():
                stock_pos = valid_pos[ids == uid]
                stock_feats = output[b][stock_pos]
                ind_node = updated[pos].unsqueeze(0).expand(stock_feats.shape[0], -1)
                gate = self.stock_gate(torch.cat([stock_feats, ind_node], dim=-1))
                output[b][stock_pos] = self.out_norm(stock_feats + gate * (ind_node - stock_feats))

        return output


class UltimateV7Model(nn.Module):
    """
    V7改进：
    - FeatureGrouper: 对5种聚合（last/sma5/sma20/vol5/vol20）做组内+跨组编码
    - Transformer: 跨股票注意力学习排序关系
    - GAT分支: 行业图注意力（可选）
    - FusionGate: 自适应融合两支
    - 多周期预测头: h1/h3/h5/h7 四个horizon

    exp-004: 移除 FiLM，所有特征平等进入 FeatureGrouper → Transformer 自己学跨特征关系
    """

    def __init__(self, input_dim, agg_groups=None, low_feat_dim=14,
                 hidden_dim=256, n_heads=8, n_layers=2,
                 n_horizons=4, n_alpha=4, use_gat=False, regime_dim=87,
                 num_industries=83, industry_emb_dim=16, dropout=0.5):
        super().__init__()
        self.use_gat = use_gat
        self.hidden_dim = hidden_dim
        self.n_horizons = n_horizons
        self.industry_emb_dim = industry_emb_dim
        self.num_industries = num_industries
        self.low_feat_dim = low_feat_dim

        if agg_groups is None:
            agg_groups = [(23, 5, 0.1), (7, 2, 0.0)]

        self.feature_grouper = FeatureGrouper(input_dim, hidden_dim, agg_groups)

        # 残差直连：原始特征透传到 Transformer，FeatureGrouper 专注学聚合间交互
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.trans_input_proj = nn.Linear(input_dim, hidden_dim)  # X→Transformer入口残差

        # 行业embedding（真实行业 + 1个未知行业）
        self.industry_embed = nn.Embedding(num_industries + 1, industry_emb_dim)
        self.industry_proj = nn.Linear(hidden_dim + industry_emb_dim, hidden_dim)

        # 排名嵌入
        self.rank_embed = nn.Embedding(512, hidden_dim)

        # GAT Block: 2层 Transformer + GAT + FusionGate，叠两次
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True
        )
        n_per_block = max(1, n_layers // 2)
        self.trans1 = nn.TransformerEncoder(enc_layer, num_layers=n_per_block)
        self.trans2 = nn.TransformerEncoder(enc_layer, num_layers=n_per_block)

        # 多头Alpha（LayerNorm 保留，Dropout 移除：transformer dropout 已足够）
        self.alpha_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_alpha)
        ])
        self.alpha_gate = nn.Sequential(
            nn.Linear(hidden_dim, n_alpha), nn.Softmax(dim=-1)
        )

        # 多周期预测头（LayerNorm 保留，Dropout 移除）
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_horizons)
        ])

        # 市场状态编码(regime_dim = 股票级风险 + 市场特征 + 可选宏观特征)
        self.regime_proj = nn.Linear(regime_dim, hidden_dim)

        # GAT分支 ×2：每轮 = 1层GATConv + 残差 + 跨行业attention + FusionGate
        if use_gat:
            for suffix in ['1', '2']:
                setattr(self, f'gat_proj_{suffix}', nn.Linear(hidden_dim, hidden_dim))
                setattr(self, f'gat_conv_{suffix}', GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.2))
                setattr(self, f'gat_dropout_{suffix}', nn.Dropout(0.2))
                setattr(self, f'gat_norm_{suffix}', nn.LayerNorm(hidden_dim))
                setattr(self, f'cross_ind_attn_{suffix}', CrossIndustryAttention(hidden_dim, num_heads=4))
                setattr(self, f'fusion_gate_{suffix}', FusionGate(hidden_dim))
        self._edge_cache = OrderedDict()
        self._edge_cache_max_size = 512

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
        """Build same-industry graph edges on GPU (native, no CPU transfer)."""
        device = industry_ids.device
        B, N = industry_ids.shape

        # 缓存key：用 tensor hash（GPU tensor bytes，比 CPU numpy 更快）
        ids_contig = industry_ids.contiguous()
        mask_contig = mask.contiguous()
        cache_key = (B, N, zlib.crc32(ids_contig.view(-1).cpu().numpy().tobytes()),
                     zlib.crc32(mask_contig.view(-1).cpu().numpy().tobytes()),
                     max_edges_per_stock, self.training)
        if cache_key in self._edge_cache:
            cached = self._edge_cache.pop(cache_key)
            self._edge_cache[cache_key] = cached
            return cached

        batch_edges = []
        for b in range(B):
            valid = mask[b]
            ids = industry_ids[b]
            # GPU: 获取有效且非unkown的行业id
            id_mask = (ids >= 0) & valid
            valid_ids = ids[id_mask]
            unique_ids = valid_ids.unique()

            edges_list = []
            for uid in unique_ids:
                ind_mask = id_mask & (ids == uid)
                idx_global = ind_mask.nonzero(as_tuple=False).squeeze(-1)
                n_ind = idx_global.shape[0]
                if n_ind < 2:
                    continue

                k = min(max_edges_per_stock, n_ind - 1)
                src_all = idx_global.repeat_interleave(k)

                if self.training:
                    # GPU 向量化随机邻居（不含自身）
                    r = torch.randint(0, n_ind - 1, (n_ind, k), device=device)
                    r = torch.where(r >= torch.arange(n_ind, device=device).unsqueeze(1), r + 1, r)
                    dst_all = idx_global[r].reshape(-1)
                else:
                    # GPU 确定性邻居（环形偏移）
                    offsets = torch.arange(1, k + 1, device=device)
                    dst_all = idx_global[(torch.arange(n_ind, device=device).unsqueeze(1) + offsets) % n_ind].reshape(-1)

                edges_list.append(torch.stack([src_all, dst_all]))

            if edges_list:
                e = torch.cat(edges_list, dim=1)
                e = torch.cat([e, e.flip(0)], dim=1)  # 无向图
            else:
                e = torch.zeros(2, 0, dtype=torch.long, device=device)
            batch_edges.append(e)

        self._edge_cache[cache_key] = batch_edges
        while len(self._edge_cache) > self._edge_cache_max_size:
            self._edge_cache.popitem(last=False)
        return batch_edges

    def _gat_forward(self, h, mask, industry_ids, trans_out=None, suffix='1'):
        """GAT forward: 1-layer GATConv + residual + cross-industry attention"""
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
            x = getattr(self, f'gat_proj_{suffix}')(h[b])
            identity = x  # GATConv 残差
            x = getattr(self, f'gat_conv_{suffix}')(x, edges)
            x = F.gelu(x)
            x = getattr(self, f'gat_dropout_{suffix}')(x)
            x = x + identity  # GATConv 残差
            x = getattr(self, f'gat_norm_{suffix}')(x)
            x = x + getattr(self, f'cross_ind_attn_{suffix}')(
                x.unsqueeze(0), industry_ids[b:b+1], mask[b:b+1]
            ).squeeze(0)  # CrossIndustryAttention 残差
            gat_out[b] = fallback[b] + x  # 整体 GAT 残差：保留 Transformer 信号
            del batch_edges, edges, x, identity
        return gat_out

    def forward(self, X, risk_cont, mask, industry_ids):
        """
        Args:
            X: (B, N, 250) = agg(129) + rank(115) + ind_rel(6)
            mask: (B, N) bool, True=valid
            industry_ids: (B, N) long, industry ID
        Returns:
            alpha_raw: (B, N) alphas
            horizon_preds: (B, N, n_horizons) preds
        """
        B, N, _ = X.shape

        # 1. 特征分组编码 + 残差直连
        h = self.feature_grouper(X) + self.input_proj(X)  # (B, N, hidden_dim)

        # 2. 行业embedding（残差注入）
        industry_ids_valid = torch.where(industry_ids >= 0, industry_ids, self.num_industries)
        industry_ids_valid = torch.clamp(industry_ids_valid, 0, self.num_industries)
        industry_emb = self.industry_embed(industry_ids_valid)
        ind_info = self.industry_proj(torch.cat([h, industry_emb], dim=-1))
        h = h + ind_info

        # 3. 排名嵌入
        rank_emb = self._build_rank_embed(X, mask)
        h = h + rank_emb

        # 4. 市场状态
        mask_f = mask.float().unsqueeze(-1)
        regime_sum = (risk_cont * mask_f).sum(dim=1)
        regime_cnt = mask_f.sum(dim=1).clamp(min=1)
        regime_avg = regime_sum / regime_cnt
        regime_h = self.regime_proj(regime_avg).unsqueeze(1)

        # 5. X→Transformer入口残差：原始特征直接透传到Transformer
        h = h + self.trans_input_proj(X)

        # 6. Transformer blocks
        if self.use_gat:
            t1 = self.trans1(h, src_key_padding_mask=~mask) + regime_h
            g1 = self._gat_forward(t1, mask, industry_ids, trans_out=t1, suffix='1')
            h1 = self.fusion_gate_1(t1, g1)
            t2 = self.trans2(h1, src_key_padding_mask=~mask) + regime_h
            g2 = self._gat_forward(t2, mask, industry_ids, trans_out=t2, suffix='2')
            h_out = self.fusion_gate_2(t2, g2)
        else:
            t1 = self.trans1(h, src_key_padding_mask=~mask) + regime_h
            h_out = self.trans2(t1, src_key_padding_mask=~mask) + regime_h

        # 7. Alpha / Horizon heads
        alphas = torch.cat([head(h_out) for head in self.alpha_heads], dim=-1)
        gate = self.alpha_gate(h_out)
        alpha_raw = (alphas * gate).sum(dim=-1)

        horizon_preds = torch.cat(
            [head(h_out) for head in self.horizon_heads], dim=-1
        )

        return alpha_raw, alphas, horizon_preds
