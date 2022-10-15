"""
Kernel Matrix Factorization (KMF) model.
"""

import numpy as np
import torch
from torch import nn


def mse(scores, pred):
    return (scores - pred).pow(2).mean().sqrt()


def get_param_squared_norms(users_emb, items_emb, users_bias, items_bias):
    emb_norms = users_emb.pow(2).sum(1).mean() + items_emb.pow(2).sum(1).mean()
    bias_norms = users_bias.pow(2).mean() + items_bias.pow(2).mean()
    return emb_norms + bias_norms


class KMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, max_score: int):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.user_bias = nn.Parameter(torch.zeros(n_users))
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        nn.init.normal_(self.user_emb.weight, 0, 0.1)
        nn.init.normal_(self.item_emb.weight, 0, 0.1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.max_score = max_score

    def forward(self, users, items):
        users_emb = self.user_emb(users)
        items_emb = self.item_emb(items)
        users_bias = self.user_bias[users]
        items_bias = self.item_bias[items]
        pred_score = self.max_score * torch.sigmoid(
            self.global_bias + items_bias + users_bias + (items_emb * users_emb).sum(1)
        )
        return pred_score, users_emb, items_emb, users_bias, items_bias
