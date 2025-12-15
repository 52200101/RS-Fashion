import torch
import torch.nn.functional as F
import numpy as np

# =========================
# USER PROFILE (CBF)
# =========================
def build_user_profile(u, user_train_items, item_feat, has_emb_t=None):
    items = user_train_items.get(u, [])
    if not items:
        return None

    items = torch.tensor(items, dtype=torch.long, device=item_feat.device)

    if has_emb_t is not None:
        mask = has_emb_t[items]
        items = items[mask]
        if items.numel() == 0:
            return None

    embs = item_feat[items]
    prof = embs.mean(dim=0)
    return F.normalize(prof, dim=0)


# =========================
# RECOMMEND (HYBRID)
# =========================
def recommend_for_user(
    u,
    user_train_items,
    user_emb_cf,
    item_emb_cf,
    item_feat,
    has_emb_t=None,
    topk=10,
    min_hist=3,
):
    num_users = user_emb_cf.shape[0]
    if u >= num_users:
        return []

    history = user_train_items.get(u, [])
    n_hist = len(history)

    # cold user → popularity fallback
    if n_hist == 0:
        scores = item_emb_cf.norm(dim=1)
        top_idx = torch.topk(scores, k=min(topk, scores.shape[0])).indices
        return [(int(i), float(scores[i])) for i in top_idx]

    # dynamic weight
    alpha_cf = 0.7 if n_hist >= min_hist else 0.3
    alpha_cb = 1.0 - alpha_cf

    # normalize
    user_emb_cf = F.normalize(user_emb_cf, dim=1)
    item_emb_cf = F.normalize(item_emb_cf, dim=1)
    item_feat = F.normalize(item_feat, dim=1)

    # CF score
    scores_cf = item_emb_cf @ user_emb_cf[u]

    # CBF score
    user_prof = build_user_profile(u, user_train_items, item_feat, has_emb_t)
    if user_prof is None:
        scores_cb = torch.zeros_like(scores_cf)
    else:
        scores_cb = item_feat @ user_prof

    # ensemble
    scores = alpha_cf * scores_cf + alpha_cb * scores_cb

    # mask seen
    if history:
        scores[torch.tensor(history, device=scores.device)] = -1e9

    if has_emb_t is not None:
        scores[~has_emb_t] = -1e9

    top_idx = torch.topk(scores, k=min(topk, scores.shape[0])).indices
    return [(int(i), float(scores[i])) for i in top_idx]


# =========================
# EVALUATE: CBF ONLY
# =========================
def evaluate_cbf_cf_idx(
    item_feat,
    user_train_items,
    user_test_items,
    has_emb_t=None,
    k=10,
):
    item_feat = F.normalize(item_feat, p=2, dim=1)

    precisions, recalls, ndcgs, hit_rates = [], [], [], []

    for u in user_test_items.keys():
        test_items = set(user_test_items.get(u, []))
        if not test_items:
            continue

        user_prof = build_user_profile(u, user_train_items, item_feat, has_emb_t)
        if user_prof is None:
            continue

        scores = item_feat @ user_prof

        train_items = set(user_train_items.get(u, []))
        if train_items:
            scores[list(train_items)] = -1e9

        if has_emb_t is not None:
            scores[~has_emb_t] = -1e9

        top_k = torch.topk(scores, k=min(k, scores.shape[0])).indices.cpu().numpy()
        hits = len(set(top_k) & test_items)

        precision = hits / min(k, len(test_items))
        recall = hits / len(test_items)

        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, iid in enumerate(top_k)
            if iid in test_items
        )
        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(test_items), k))
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0

        hit = 1.0 if hits > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hit_rates.append(hit)

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "Users_eval": len(precisions),
    }


# =========================
# EVALUATE: ENSEMBLE CF + CBF
# =========================
def evaluate_cf_cbf_ensemble(
    user_emb_cf,
    item_emb_cf,
    item_feat,
    user_train_items,
    user_test_items,
    has_emb_t=None,
    alpha_cf=0.7,
    alpha_cb=0.3,
    k=10,
):
    user_emb_cf = F.normalize(user_emb_cf, dim=1)
    item_emb_cf = F.normalize(item_emb_cf, dim=1)
    item_feat = F.normalize(item_feat, dim=1)

    precisions, recalls, ndcgs, hit_rates = [], [], [], []

    num_users = user_emb_cf.shape[0]

    for u in user_test_items.keys():
        # 🔑 FIX QUAN TRỌNG: bỏ user không có embedding
        if u >= num_users:
            continue

        test_items = set(user_test_items.get(u, []))
        if not test_items:
            continue

        scores_cf = item_emb_cf @ user_emb_cf[u]

        user_prof = build_user_profile(u, user_train_items, item_feat, has_emb_t)
        if user_prof is None:
            scores_cb = torch.zeros_like(scores_cf)
        else:
            scores_cb = item_feat @ user_prof

        scores = alpha_cf * scores_cf + alpha_cb * scores_cb

        train_items = set(user_train_items.get(u, []))
        if train_items:
            scores[list(train_items)] = -1e9

        if has_emb_t is not None:
            scores[~has_emb_t] = -1e9

        top_k = torch.topk(scores, k=min(k, scores.shape[0])).indices.cpu().numpy()
        hits = len(set(top_k) & test_items)

        precision = hits / min(k, len(test_items))
        recall = hits / len(test_items)

        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, iid in enumerate(top_k)
            if iid in test_items
        )
        idcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(test_items), k))
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0

        hit = 1.0 if hits > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hit_rates.append(hit)

    return {
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"HitRate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "Users_eval": len(precisions),
    }
