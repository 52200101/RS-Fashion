import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=128, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, norm_adj):
        all_embs = []
        x = self.embedding.weight
        all_embs.append(x)

        for _ in range(self.n_layers):
            x = torch.sparse.mm(norm_adj, x)
            x = F.normalize(x, dim=1)
            all_embs.append(x)

        embs = torch.stack(all_embs, dim=1)
        out = torch.mean(embs, dim=1)
        out = F.normalize(out, dim=1)
        return out
