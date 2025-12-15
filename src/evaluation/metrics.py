import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

def l2n(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def evaluate_embeddings(user_emb, item_emb, user_train_items, user_test_items, k=10):
    U = l2n(user_emb.cpu())
    I = l2n(item_emb.cpu())
    I_np = I.numpy()

    precisions, recalls, ndcgs, hit_rates = [], [], [], []

    num_users = user_emb.shape[0]

    for u in user_test_items.keys():
        # bỏ user không có embedding
        if u >= num_users:
            continue
        test_items = set(user_test_items[u])
        if not test_items:
            continue

        scores = cosine_similarity(
            U[u].unsqueeze(0).numpy(), I_np
        )[0]

        # # mask train items
        # train_items = set(user_train_items.get(u, []))
        # if train_items:
        #     scores[list(train_items)] = -1e9

        top_k = np.argsort(-scores)[:k]
        hits = len(set(top_k) & test_items)

        precision = hits / min(k, len(test_items))
        recall = hits / len(test_items)

        dcg = sum(
            1 / np.log2(i + 2)
            for i, iid in enumerate(top_k)
            if iid in test_items
        )
        idcg = sum(
            1 / np.log2(i + 2)
            for i in range(min(len(test_items), k))
        )
        ndcg = dcg / idcg if idcg > 0 else 0.0

        hit = 1.0 if hits > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
        hit_rates.append(hit)

    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"HitRate@{k}": float(np.mean(hit_rates)),
        "Users_eval": len(precisions),
    }
