# model_gat.py 鈥?GAT 浜т笟閾惧浘娉ㄦ剰鍔涚綉缁?import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ==================== 琛屼笟鍥捐氨鏋勫缓 ====================
def build_industry_graph(industry_dict, codes):
    """
    鍩轰簬琛屼笟鍒嗙被鏋勯€犻偦鎺ョ煩闃?
    Args:
        industry_dict: {code: industry_name} 鏄犲皠
        codes: 褰撳墠鎴?潰鑲＄エ浠ｇ爜鍒楄〃
    Returns:
        edge_index: torch.LongTensor shape (2, num_edges)
        edge_weight: torch.FloatTensor shape (num_edges,)
    """
    code_to_idx = {code: i for i, code in enumerate(codes)}
    edges_src, edges_dst, edges_w = [], [], []

    # same industry connection
    for code_a in codes:
        idx_a = code_to_idx[code_a]
        ind_a = industry_dict.get(code_a, 'Unknown')
        if ind_a is None or ind_a == 'Unknown':
            continue
        for code_b in codes:
            idx_b = code_to_idx[code_b]
            if idx_a <= idx_b:
                continue
            ind_b = industry_dict.get(code_b, 'Unknown')
            if ind_a == ind_b:
                edges_src.append(idx_a)
                edges_dst.append(idx_b)
                edges_w.append(1.0)
                # 鍙屽悜
                edges_src.append(idx_b)
                edges_dst.append(idx_a)
                edges_w.append(1.0)

    if not edges_src:
        # 鏃犺?涓氫俊鎭?椂锛屾瀯閫犲叏杩炴帴鑷?幆
        N = len(codes)
        edge_index = torch.arange(N, dtype=torch.long).repeat(2, 1)
        edge_weight = torch.ones(N)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_weight = torch.tensor(edges_w, dtype=torch.float)

    return edge_index, edge_weight


def build_correlation_edges(X, top_k=5):
    """
    鍩轰簬鐗瑰緛鐩镐技搴﹁ˉ鍏呰法琛屼笟杩炶竟锛堝彲閫夛級

    Args:
        X: (N, F) 鐗瑰緛鐭╅樀
        top_k: 姣忓彧鑲＄エ杩炴帴鏈€鐩镐技鐨?K 涓?偦灞?    Returns:
        edge_index, edge_weight
    """
    N = X.shape[0]
    # 浣欏鸡鐩镐技搴?    X_norm = X / (torch.norm(X, dim=1, keepdim=True) + 1e-8)
    sim = X_norm @ X_norm.T  # (N, N)

    edges_src, edges_dst, edges_w = [], [], []
    for i in range(N):
        sim_i = sim[i].clone()
        sim_i[i] = -1  # 鎺掗櫎鑷?幆
        top_idx = torch.topk(sim_i, min(top_k, N - 1)).indices
        for j in top_idx:
            edges_src.append(i)
            edges_dst.append(int(j))
            edges_w.append(float(sim_i[j]))

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_weight = torch.tensor(edges_w, dtype=torch.float)
    return edge_index, edge_weight


# ==================== GAT 棰勬祴妯″瀷 ====================
class GATPredictor(nn.Module):
    """GAT frontend predictor"""

    def __init__(self, input_dim, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # GAT 鍗风Н灞?        self.gat_layers = nn.ModuleList()
        for i in range(n_layers):
            in_ch = hidden_dim if i == 0 else hidden_dim * n_heads
            out_ch = hidden_dim
            self.gat_layers.append(
                GATConv(in_ch, out_ch, heads=n_heads, dropout=dropout, concat=True)
            )

        # 杈撳嚭澶?        final_dim = hidden_dim * n_heads
        self.pred_head = nn.Sequential(
            nn.Linear(final_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, X, edge_index, edge_weight=None, batch=None):
        """
        X: (N, F) feature matrix
        edge_index: (2, E) edge indices
        edge_weight: (E,) edge weights
        batch: (N,) batch index
        """
        h = self.input_proj(X)
        h_res = h  # 娈嬪樊杩炴帴璧风偣

        for gat in self.gat_layers:
            h_new = gat(h, edge_index, edge_weight)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            h = h_new

        # 鎷兼帴鍘熷?鐗瑰緛鎶曞奖 + GAT杈撳嚭
        h_out = torch.cat([h, h_res], dim=-1)
        pred = self.pred_head(h_out).squeeze(-1)
        return pred


# ==================== GAT 鎹熷け鍑芥暟 ====================
def gat_rank_loss(pred, target):
    """Spearman rank loss"""
    if pred.numel() < 5:
        return torch.tensor(0.0, device=pred.device)
    pred_z = (pred - pred.mean()) / (pred.std() + 1e-8)
    target_z = (target - target.mean()) / (target.std() + 1e-8)
    return -(pred_z * target_z).mean()


# ==================== 鐙?珛娴嬭瘯 ====================
if __name__ == "__main__":
    print("=== GAT 妯″瀷娴嬭瘯 ===\n")

    # 妯℃嫙鏁版嵁
    N = 100
    F = 74
    X = torch.randn(N, F)

    # 妯℃嫙琛屼笟
    codes = [f"{600000 + i}" for i in range(N)]
    industries = ['閾惰?'] * 20 + ['鐧介厭'] * 15 + ['鍖昏嵂'] * 25 + ['鍦颁骇'] * 20 + ['鐢靛姏'] * 20
    industry_dict = {code: industries[i] for i, code in enumerate(codes)}

    print(f"industry graph: {N} nodes, {edge_index.shape[1]} edges")
    print(f"industry graph: {N} nodes, {edge_index.shape[1]} edges")

    # 妯″瀷鍓嶅悜
    model = GATPredictor(input_dim=F, hidden_dim=128, n_heads=4, n_layers=2)
    pred = model(X, edge_index, edge_weight)
    print(f"棰勬祴杈撳嚭: {pred.shape}")

    # 鎹熷け璁＄畻
    y = torch.randn(N)
    loss = gat_rank_loss(pred, y)
    print(f"Loss: {loss.item():.4f}")

    # 鍙傛暟閲?    n_params = sum(p.numel() for p in model.parameters())
    print(f"鍙傛暟鎬婚噺: {n_params:,}")
