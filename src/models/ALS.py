import torch

class ALS:
    def __init__(self, user_emb, item_emb):
        self.user_emb = user_emb
        self.item_emb = item_emb

    def recommend(self, u, topk=10):
        scores = self.item_emb @ self.user_emb[u]
        topk_idx = torch.topk(scores, k=topk).indices
        return [(int(i), float(scores[i])) for i in topk_idx]
