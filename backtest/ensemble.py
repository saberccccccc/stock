# ensemble.py - Stacking 集成（LightGBM预测 + Transformer特征）
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

from core.config import DataConfig
from data.pipeline import build_cross_section_dataset, N_AGGS, INDUSTRY_REL_FEATURES
from core.model import UltimateV7Model
from core.train_utils import get_regime_dim


# ==================== 时序交叉验证预测 ====================
def cv_predict_lgb(train_samples, val_samples, horizon_list=(1, 3, 5, 10),
                   n_folds=5, model_dir="models_multi_cv"):
    """OOZE prediction: CV for training set, full model prediction for validation set"""
    os.makedirs(model_dir, exist_ok=True)
    all_samples = train_samples + val_samples
    n_train = len(train_samples)
    n = len(all_samples)
    fold_size = max(1, n_train // n_folds)

    preds_by_h = {h: [None] * n for h in horizon_list}
    y_by_h = {h: np.zeros(n) for h in horizon_list}
    params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.7,
        'bagging_fraction': 0.7, 'bagging_freq': 5, 'lambda_l1': 0.1,
        'lambda_l2': 0.1, 'min_data_in_leaf': 20, 'max_depth': -1,
        'verbose': -1, 'num_threads': 4,
    }

    def fit_model(indices, h):
        X_train_list, y_train_list = [], []
        for ti in indices:
            s = all_samples[ti]
            if s['y_seq'].shape[1] < h:
                continue
            y_vals = s['y_seq'][:, h - 1]
            if np.all(np.isnan(y_vals)):
                continue
            X_train_list.append(s['X'])
            y_train_list.append(y_vals)
        if not X_train_list:
            return None
        dtrain = lgb.Dataset(np.vstack(X_train_list), label=np.hstack(y_train_list))
        return lgb.train(params, dtrain, num_boost_round=200)

    for h in horizon_list:
        for fold in range(n_folds):
            start_val = fold * fold_size
            end_val = min(start_val + fold_size, n_train) if fold < n_folds - 1 else n_train
            val_idx = list(range(start_val, end_val))
            train_idx = list(range(0, start_val))
            if not train_idx or not val_idx:
                continue
            model = fit_model(train_idx, h)
            if model is None:
                continue
            for vi in val_idx:
                s = all_samples[vi]
                if s['y_seq'].shape[1] >= h:
                    preds_by_h[h][vi] = model.predict(s['X'])
                    y_by_h[h][vi] = np.nanmean(s['y_seq'][:, h - 1])

        final_model = fit_model(list(range(n_train)), h)
        if final_model is None:
            continue
        for vi in range(n_train, n):
            s = all_samples[vi]
            if s['y_seq'].shape[1] >= h:
                preds_by_h[h][vi] = final_model.predict(s['X'])
                y_by_h[h][vi] = np.nanmean(s['y_seq'][:, h - 1])

    return preds_by_h, y_by_h


# ==================== Stacking 数据集====================
class StackingDataset(Dataset):
    """Stack original features + LGB predictions into 2nd-level input"""

    def __init__(self, samples, lgb_preds_dict, sample_offset=0):
        """
        samples: 截面样本列表
        lgb_preds_dict: {horizon: (n_samples,) array} - CV预测
        sample_offset: samples在train+val合并列表中的起始位置
        """
        self.X_list = []
        self.y_list = []
        self.y_seq_list = []
        self.risk_list = []
        self.codes_list = []
        self.industry_ids_list = []

        for i, s in enumerate(samples):
            N, F = s['X'].shape
            # 将LGB预测作为额外特征拼入
            extra_feats = np.zeros((N, len(lgb_preds_dict)), dtype=np.float32)
            global_i = sample_offset + i
            for j, h in enumerate(sorted(lgb_preds_dict.keys())):
                if global_i < len(lgb_preds_dict[h]) and lgb_preds_dict[h][global_i] is not None and len(lgb_preds_dict[h][global_i]) == N:
                    extra_feats[:, j] = lgb_preds_dict[h][global_i]
                else:
                    extra_feats[:, j] = 0.0

            X_stacked = np.concatenate([s['X'], extra_feats], axis=1)
            self.X_list.append(X_stacked.astype(np.float32))
            self.y_list.append(s['y'].astype(np.float32))
            self.y_seq_list.append(s['y_seq'].astype(np.float32))
            self.risk_list.append(s['risk'].astype(np.float32))
            self.codes_list.append(s['codes'])
            self.industry_ids_list.append(s['industry_ids'].astype(np.int64))

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        return {
            'X': torch.from_numpy(self.X_list[idx]),
            'y': torch.from_numpy(self.y_list[idx]),
            'y_seq': torch.from_numpy(self.y_seq_list[idx]),
            'risk': torch.from_numpy(self.risk_list[idx]),
            'industry_ids': torch.from_numpy(self.industry_ids_list[idx]),
        }


# ==================== Collate ====================
def collate_stacking(batch):
    B = len(batch)
    max_N = max(item['X'].shape[0] for item in batch)
    F = batch[0]['X'].shape[1]
    H = batch[0]['y_seq'].shape[1]
    R = batch[0]['risk'].shape[1]

    X = torch.zeros(B, max_N, F)
    y = torch.zeros(B, max_N)
    y_seq = torch.zeros(B, max_N, H)
    risk = torch.zeros(B, max_N, R)
    industry_ids = torch.full((B, max_N), -1, dtype=torch.long)
    mask = torch.zeros(B, max_N, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item['X'].shape[0]
        X[i, :n] = item['X']
        y[i, :n] = item['y']
        y_seq[i, :n] = item['y_seq']
        risk[i, :n] = item['risk']
        industry_ids[i, :n] = item['industry_ids']
        mask[i, :n] = 1
    return {'X': X, 'y': y, 'y_seq': y_seq, 'risk': risk, 'industry_ids': industry_ids, 'mask': mask}


# ==================== 加权融合 ====================
def blend_predictions(lgb_preds, dl_alpha, weights=(0.5, 0.5)):
    """
    融合LightGBM和深度学习预测

    Args:
        lgb_preds: (N,) LightGBM融合alpha
        dl_alpha: (N,) Transformer/GAT alpha
        weights: (w_lgb, w_dl) 融合权重
    Returns:
        blended: (N,) 融合后alpha
    """
    w_lgb, w_dl = weights
    blended = w_lgb * lgb_preds + w_dl * dl_alpha
    # 截面标准化
    if len(blended) > 1:
        blended = (blended - np.mean(blended)) / (np.std(blended) + 1e-8)
    return np.tanh(blended)


# ==================== 训练入口 ====================
def train_stacking(train_samples, val_samples, input_dim, horizon=10, epochs=15,
                   lr=3e-4, batch_size=8, cfg=None):
    """训练Stacking二级模型"""
    cfg = cfg or DataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先跑LGB CV获取预测
    print("Step 1: LightGBM交叉验证...")
    lgb_preds, _ = cv_predict_lgb(train_samples, val_samples)

    # 构建Stacking数据集（当前简化为直接增加占位特征）
    n_lgb_feats = len(lgb_preds)
    stacking_dim = input_dim + n_lgb_feats
    print(f"Stacking输入维度: {input_dim} + {n_lgb_feats} = {stacking_dim}")

    train_ds = StackingDataset(train_samples, lgb_preds, sample_offset=0)
    val_ds = StackingDataset(val_samples, lgb_preds, sample_offset=len(train_samples))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_stacking)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_stacking)

    print("Step 2: 训练Transformer二级模型...")
    base_feat_dim = (input_dim - len(INDUSTRY_REL_FEATURES)) // 2 // N_AGGS
    regime_dim = get_regime_dim(cfg)
    num_industries = train_samples[0]['risk'].shape[1] - regime_dim
    model = UltimateV7Model(
        stacking_dim, base_feat_dim, n_aggs=N_AGGS,
        hidden_dim=256, n_heads=8, n_layers=4,
        n_horizons=horizon, n_alpha=4, use_gat=False,
        regime_dim=regime_dim, num_industries=num_industries
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            risk = batch['risk'].to(device)
            industry_ids = batch['industry_ids'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            alpha_raw, _, _ = model(X, risk[:, :, :regime_dim], mask, industry_ids)

            # Spearman loss
            pred = alpha_raw[mask]
            target = y[mask]
            if pred.numel() > 10:
                pred_z = (pred - pred.mean()) / (pred.std() + 1e-8)
                target_z = (target - target.mean()) / (target.std() + 1e-8)
                loss = -(pred_z * target_z).mean()
            else:
                loss = torch.tensor(0.0, device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_ic = []
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                risk = batch['risk'].to(device)
                industry_ids = batch['industry_ids'].to(device)
                mask = batch['mask'].to(device)

                alpha_raw, _, _ = model(X, risk[:, :, :regime_dim], mask, industry_ids)
                for b in range(X.shape[0]):
                    m = mask[b]
                    if m.sum() > 10:
                        pred_np = alpha_raw[b][m].cpu().numpy()
                        y_np = y[b][m].cpu().numpy()
                        ic, _ = spearmanr(pred_np, y_np)
                        if np.isfinite(ic):
                            val_ic.append(ic)

        mean_ic = np.mean(val_ic) if val_ic else 0.0
        scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val IC: {mean_ic:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_ic': mean_ic,
            }, "stacking_best.pt")
            print(f"")

    return model


if __name__ == "__main__":
    print("=== Stacking集成测试 ===\n")

    cfg = DataConfig()
    cfg.use_technical_features = True
    cfg.use_macro_features = True
    cfg.min_stocks_per_time = 30
    cfg.target_horizon = 5
    cfg.seq_len = 40
    cfg.max_horizon = 10

    print("加载数据...")
    train_samples, val_samples = build_cross_section_dataset(cfg, use_cache=True)
    input_dim = train_samples[0]['X'].shape[1]
    print(f"Input dim: {input_dim}, 训练样本: {len(train_samples)}, 验证样本: {len(val_samples)}")

    print("\n开始Stacking训练...")
    model = train_stacking(train_samples, val_samples, input_dim, horizon=10,
                           epochs=5, batch_size=8, cfg=cfg)
