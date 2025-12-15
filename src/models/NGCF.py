import torch
import torch.nn as nn
import torch.nn.functional as F

class NGCF(nn.Module):
    def __init__(self, n_users, n_items, dim=128, layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        self.W1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.W2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, norm_adj):
        x = self.embedding.weight
        embs = [x]

        for k in range(len(self.W1)):
            side = torch.sparse.mm(norm_adj, x)
            x = F.leaky_relu(self.W1[k](side + x) + self.W2[k](side * x))
            x = self.dropout(x)
            x = F.normalize(x, dim=1)
            embs.append(x)

        return torch.cat(embs, dim=1)
