# model.py 鈥?UltimateV7 鎴?潰Alpha妯″瀷
# V7鏀硅繘锛氱壒寰佸垎缁勬椂搴忕紪鐮?+ 璺ㄨ偂绁═ransformer + 鍙?€塆AT鍒嗘敮 + 澶氬懆鏈熷ご
# V9鏀硅繘锛氬ご閮ㄥ垎鏀疍ropout + 鏇村己姝ｅ垯鍖?
# GAT瀹炵幇 (2026-05-12): 琛屼笟鍥炬敞鎰忓姏 + FusionGate铻嶅悎
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FeatureGrouper(nn.Module):
    """鎸夎仛鍚堢被鍨嬪垎缁勭殑鐗瑰緛缂栫爜鍣?細last/sma5/sma20/vol5/vol20 鈫?鏃跺簭鎰熺煡embedding"""

    def __init__(self, input_dim, hidden_dim, base_feat_dim, n_aggs=5):
        super().__init__()
        self.base_feat_dim = base_feat_dim
        self.n_aggs = n_aggs
        self.agg_feat_dim = base_feat_dim * n_aggs

        # 纭?繚 per_agg_dim 鑳借?娉ㄦ剰鍔涘ご鏁版暣闄?
        raw_dim = hidden_dim // n_aggs  # e.g. 256//5 = 51
        self.per_agg_dim = 48           # 48 鍙?? 2/3/4 鏁撮櫎
        self.num_agg_heads = 2          # 48/2 = 24 per head
        agg_total = self.per_agg_dim * n_aggs  # 48*5 = 240
        self.extra_dim = hidden_dim - agg_total  # 256-240 = 16

        # 缁勫唴鎶曞奖锛氭瘡涓?仛鍚堢被鍨嬬嫭绔嬫姇褰?
        self.agg_proj = nn.ModuleList([
            nn.Linear(base_feat_dim, self.per_agg_dim)
            for _ in range(n_aggs)
        ])
        # 璺ㄨ仛鍚堢被鍨嬬壒寰佷氦浜掞紙瀛︿範涓嶅悓鏃堕棿灏哄害鐨勫叧绯伙級
        self.cross_agg_attn = nn.MultiheadAttention(
            embed_dim=self.per_agg_dim, num_heads=self.num_agg_heads,
            batch_first=True, dropout=0.3
        )
        # 棰濆?鐨?skip-projection 鐢ㄤ簬闈炶仛鍚堢壒寰?
        self.extra_proj = nn.Linear(
            input_dim - self.agg_feat_dim, self.extra_dim
        )

    def forward(self, x):
        # x: (B, N, F_total)
        B, N, F = x.shape
        agg_feat = x[..., :self.agg_feat_dim]
        extra_feat = x[..., self.agg_feat_dim:]

        # 姣忎釜鑱氬悎绫诲瀷鐙?珛缂栫爜
        agg_parts = []
        for i in range(self.n_aggs):
            chunk = agg_feat[..., i * self.base_feat_dim:(i + 1) * self.base_feat_dim]
            proj = self.agg_proj[i](chunk)  # (B, N, h/n_aggs)
            agg_parts.append(proj)

        # 璺ㄨ仛鍚堢被鍨嬫敞鎰忓姏锛氭妸 n_aggs 褰撲綔"搴忓垪"缁村害
        # (B*N, n_aggs, h/n_aggs)
        stacked = torch.stack(agg_parts, dim=1).reshape(B * N, self.n_aggs, -1)
        attn_out, _ = self.cross_agg_attn(stacked, stacked, stacked)
        attn_out = attn_out.reshape(B, N, self.n_aggs, -1)
        agg_out = attn_out.reshape(B, N, -1)  # (B, N, hidden_dim')

        # 棰濆?鐗瑰緛锛坮ank + industry_relative锛夌洿鎺ユ姇褰卞苟鎷兼帴
        extra_out = self.extra_proj(extra_feat)
        return torch.cat([agg_out, extra_out], dim=-1)  # (B, N, hidden_dim)


class FusionGate(nn.Module):
    """鑷?€傚簲铻嶅悎 Transformer 鍜?GAT 鍒嗘敮"""

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
    V7鏀硅繘锛?
    - FeatureGrouper: 瀵?绉嶈仛鍚堬紙last/sma5/sma20/vol5/vol20锛夊仛缁勫唴+璺ㄧ粍缂栫爜
    - Transformer: 璺ㄨ偂绁ㄦ敞鎰忓姏瀛︿範鎺掑簭鍏崇郴
    - GAT鍒嗘敮: 琛屼笟鍥炬敞鎰忓姏锛堝彲閫夛級
    - FusionGate: 鑷?€傚簲铻嶅悎涓ゆ敮
    - 澶氬懆鏈熼?娴嬪ご: h1/h3/h5/h7 鍥涗釜horizon
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

        # 鐗瑰緛鍒嗙粍缂栫爜
        self.feature_grouper = FeatureGrouper(
            input_dim, hidden_dim, base_feat_dim, n_aggs
        )

        # 琛屼笟embedding锛堢湡瀹炶?涓?+ 1涓?湭鐭ヨ?涓氾級
        self.industry_embed = nn.Embedding(num_industries + 1, industry_emb_dim)
        self.industry_proj = nn.Linear(hidden_dim + industry_emb_dim, hidden_dim)

        # 鎺掑悕宓屽叆
        self.rank_embed = nn.Embedding(512, hidden_dim)

        # Transformer 缂栫爜鍣?紙璺ㄨ偂绁ㄦ敞鎰忓姏锛?
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.35, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 澶氬ごAlpha
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

        # 澶氬懆鏈熼?娴嬪ご
        self.horizon_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(n_horizons)
        ])

        # 甯傚満鐘舵€佺紪鐮?(regime_dim = 鑲＄エ绾ч?闄?+ 甯傚満鐗瑰緛 + 鍙?€夊畯瑙傜壒寰?
        self.regime_proj = nn.Linear(regime_dim, hidden_dim)

        # GAT鍒嗘敮锛?灞侴ATConv锛岃?涓氬叏杩炴帴鍥?
        if use_gat:
            self.gat_conv = GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.3)
            self.gat_proj = nn.Linear(hidden_dim, hidden_dim)
            self.gat_norm = nn.LayerNorm(hidden_dim)
            self.fusion_gate = FusionGate(hidden_dim)
        # eval杈圭储寮曠紦瀛橈紙楠岃瘉闆嗕笉鍙橀噺锛屾瘡epoch鍙?渶绠椾竴娆★級
        self._eval_edge_cache = {}

    def _build_rank_embed(self, X, mask):
        """鍩轰簬绗?竴缁寸壒寰佹瀯寤烘帓鍚嶅祵鍏?"""
        B, N, _ = X.shape
        # 鍦ㄦ湁鏁堣偂绁ㄥ唴鎺掑悕
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
        """Build industry graph edges, CPU execution, O(n*k) random sampling
        eval mode: edge cache enabled (validation set unchanged, computed once per epoch)
        Returns: list of (2, E) edge_index, one per batch element
        """
        device = industry_ids.device
        ids_cpu = industry_ids.cpu()
        mask_cpu = mask.cpu()

        # eval妯″紡锛氱敤industry_ids鍐呭?鍋氱紦瀛榢ey锛岄伩鍏嶆瘡epoch閲嶅?鏋勫缓
        if not self.training:
            ids_bytes = ids_cpu.numpy().tobytes()
            mask_bytes = mask_cpu.numpy().tobytes()
            cache_key = (ids_bytes, mask_bytes)
            if cache_key in self._eval_edge_cache:
                return [e.to(device) for e in self._eval_edge_cache[cache_key]]

        B, N = ids_cpu.shape
        batch_edges = []
        for b in range(B):
            valid = mask_cpu[b]
            ids = ids_cpu[b]
            unique_ids = ids[(ids >= 0) & valid].unique()
            edges_list = []
            for uid in unique_ids:
                mask_uid = (ids == uid) & valid
                idx_global = mask_uid.nonzero(as_tuple=True)[0]
                n_ind = len(idx_global)
                if n_ind < 2:
                    continue
                k = min(max_edges_per_stock, n_ind - 1)
                # O(n*k) 閲囨牱锛氭瘡鍙?偂绁ㄨ繛鎺?涓?殢鏈哄悓琛?
                rand_idx = torch.randperm(n_ind)
                src = rand_idx[:, None].expand(n_ind, k).reshape(-1)
                dst = torch.stack([
                    rand_idx[torch.randperm(n_ind)][:k]
                    for _ in range(n_ind)
                ]).reshape(-1)
                keep = src != dst
                src_global = idx_global[src[keep]]
                dst_global = idx_global[dst[keep]]
                edges_list.append(torch.stack([src_global, dst_global]))
            if edges_list:
                e = torch.cat(edges_list, dim=1)
                e = torch.cat([e, e.flip(0)], dim=1)
            else:
                e = torch.zeros(2, 0, dtype=torch.long)
            batch_edges.append(e)

        if not self.training:
            # 瀛樺偍CPU鐗堟湰锛宑ache hit鏃跺欢杩?to(device)
            self._eval_edge_cache[cache_key] = batch_edges
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

        # 1. 鐗瑰緛鍒嗙粍缂栫爜
        h = self.feature_grouper(X)  # (B, N, hidden_dim)

        # 2. 琛屼笟embedding鎷兼帴骞舵姇褰?
        # 灏?1鏄犲皠鍒皀um_industries锛堜綔涓?鏈?煡琛屼笟"锛?
        industry_ids_valid = torch.where(industry_ids >= 0, industry_ids, self.num_industries)
        industry_ids_valid = torch.clamp(industry_ids_valid, 0, self.num_industries)
        industry_emb = self.industry_embed(industry_ids_valid)  # (B, N, industry_emb_dim)
        h = torch.cat([h, industry_emb], dim=-1)  # (B, N, hidden_dim + industry_emb_dim)
        h = self.industry_proj(h)  # (B, N, hidden_dim)

        # 3. 鎺掑悕宓屽叆
        rank_emb = self._build_rank_embed(X, mask)
        h = h + rank_emb

        # 4. 甯傚満鐘舵€?
        mask_f = mask.float().unsqueeze(-1)
        regime_sum = (risk_cont * mask_f).sum(dim=1)
        regime_cnt = mask_f.sum(dim=1).clamp(min=1)
        regime_avg = regime_sum / regime_cnt
        regime_h = self.regime_proj(regime_avg).unsqueeze(1)

        # 4. Transformer锛堣法鑲＄エ娉ㄦ剰鍔涳級
        trans_out = self.transformer(h, src_key_padding_mask=~mask)
        trans_out = trans_out + regime_h

        # 5. GAT鍒嗘敮锛氳?涓氬浘娉ㄦ剰鍔涗紶鎾?
        if self.use_gat:
            gat_out = self._gat_forward(h, mask, industry_ids, trans_out=trans_out)
            h_out = self.fusion_gate(trans_out, gat_out)
        else:
            h_out = trans_out

        # 6. Alpha 澶?
        alphas = torch.cat([head(h_out) for head in self.alpha_heads], dim=-1)
        gate = self.alpha_gate(h_out)
        alpha_raw = (alphas * gate).sum(dim=-1)

        # 7. 澶氬懆鏈熼?娴嬪ご
        horizon_preds = torch.cat(
            [head(h_out) for head in self.horizon_heads], dim=-1
        )  # (B, N, n_horizons)

        return alpha_raw, alphas, horizon_preds
