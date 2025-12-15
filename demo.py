import os
import pickle
import torch
import numpy as np

# ========================
# CONFIG
# ========================
DEVICE = "cpu"
MODE = "full"   # "full" | "filter"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# IMPORT
# ========================
from src.models.CBF import (
    evaluate_cbf_cf_idx,
    evaluate_cf_cbf_ensemble,
)
from src.evaluation.metrics import evaluate_embeddings
from src.utils.load_model import load_cf_model
from src.models.LightGCN import LightGCN
from src.models.NGCF import NGCF

# ========================
# LOAD DATA
# ========================
print("=> Loading processed data")

with open(os.path.join(ROOT_DIR, "data", "processed", f"user_train_items_{MODE}.pkl"), "rb") as f:
    user_train_items = pickle.load(f)

with open(os.path.join(ROOT_DIR, "data", "processed", f"user_test_items_{MODE}.pkl"), "rb") as f:
    user_test_items = pickle.load(f)

with open(os.path.join(ROOT_DIR, "data", "processed", "outfit_embeddings.pkl"), "rb") as f:
    outfit_embeddings = pickle.load(f)

with open(os.path.join(ROOT_DIR, "data", "processed", "oid_from_item.pkl"), "rb") as f:
    oid_from_item = pickle.load(f)

print(f"  - users: {len(user_train_items)}")
print(f"  - outfits with CLIP embedding: {len(outfit_embeddings)}")

# ========================
# LOAD ITEM CONTENT (CLIP)
# ========================
print("=> Building item content embedding")

# dùng LightGCN để lấy num_items
ckpt_tmp = torch.load(
    os.path.join(ROOT_DIR, "src", "saved_models", f"lightgcn_{MODE}.pt"),
    map_location=DEVICE
)
num_items = ckpt_tmp["item_emb"].shape[0]
feat_dim = next(iter(outfit_embeddings.values())).shape[0]

item_feat = torch.zeros((num_items, feat_dim), dtype=torch.float32)

for item_id, outfit_id in oid_from_item.items():
    if item_id < num_items:
        v = outfit_embeddings.get(outfit_id)
        if v is not None:
            item_feat[item_id] = torch.from_numpy(v)

item_feat = item_feat.to(DEVICE)

# ========================
# LOAD MODELS
# ========================
print("=> Loading models")

# ===== NGCF =====
ngcf_model, user_emb_ngcf, item_emb_ngcf, _ = load_cf_model(
    f"ngcf_{MODE}",
    ModelClass=NGCF,
    device=DEVICE
)


# ===== LightGCN =====
lg_model, user_emb_lg, item_emb_lg, _ = load_cf_model(
    f"lightgcn_{MODE}",
    ModelClass=LightGCN,
    device=DEVICE
)


# ===== ALS =====
_, user_emb_als, item_emb_als, _ = load_cf_model(
    f"als_{MODE}",
    ModelClass=None,
    device=DEVICE
)


# ========================
# EVALUATE
# ========================
print("=> Evaluating models")

results = {}

# ---- CF only ----
results["NGCF"] = evaluate_embeddings(
    user_emb_ngcf, item_emb_ngcf,
    user_train_items, user_test_items,
    k=10
)

results["LightGCN"] = evaluate_embeddings(
    user_emb_lg, item_emb_lg,
    user_train_items, user_test_items,
    k=10
)

results["ALS"] = evaluate_embeddings(
    user_emb_als, item_emb_als,
    user_train_items, user_test_items,
    k=10
)

# ---- CBF only ----
results["CBF"] = evaluate_cbf_cf_idx(
    item_feat,
    user_train_items,
    user_test_items,
    has_emb_t=None,
    k=10
)

# ---- Ensemble ----
results["Ensemble NGCF + CBF"] = evaluate_cf_cbf_ensemble(
    user_emb_ngcf, item_emb_ngcf,
    item_feat,
    user_train_items, user_test_items,
    alpha_cf=0.7, alpha_cb=0.3,
    k=10
)

results["Ensemble LightGCN + CBF"] = evaluate_cf_cbf_ensemble(
    user_emb_lg, item_emb_lg,
    item_feat,
    user_train_items, user_test_items,
    alpha_cf=0.7, alpha_cb=0.3,
    k=10
)

results["Ensemble ALS + CBF"] = evaluate_cf_cbf_ensemble(
    user_emb_als, item_emb_als,
    item_feat,
    user_train_items, user_test_items,
    alpha_cf=0.7, alpha_cb=0.3,
    k=10
)

# ========================
# PRINT TABLE
# ========================
print("\n=== METRIC SUMMARY (@10) ===")
print(f"{'Model':30s} {'Precision@10':>12s} {'Recall@10':>12s} {'NDCG@10':>12s} {'HitRate@10':>12s}")

for model, m in results.items():
    print(
        f"{model:30s} "
        f"{m['Precision@10']:12.5f} "
        f"{m['Recall@10']:12.5f} "
        f"{m['NDCG@10']:12.5f} "
        f"{m['HitRate@10']:12.5f}"
    )
