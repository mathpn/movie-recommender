"""
Kernel Matrix Factorization (KMF) model.
"""

import asyncpg
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from app.db.postgres import (delete_all_movie_vector_bias,
                             delete_all_user_vector_bias, sample_ratings,
                             write_bulk_movie_vector_bias,
                             write_bulk_user_vector_bias)
from app.models import VectorBias


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


async def write_model_data(embedding, biases, mapping, callback, pool, chunk_size=2048):
    chunk = []
    for user, i in mapping.items():
        vector = embedding.weight[i, :].detach().tolist()
        bias = biases[i].detach().item()
        vector_bias = VectorBias(vector, bias)
        chunk.append((vector_bias, user))
        if len(chunk) >= chunk_size or i == len(mapping) - 1:
            await callback(pool, chunk)
            chunk = []


async def fetch_data(pool: asyncpg.Pool, proportion: float, test_split: float = 0.2):
    users, movies, ratings = [], [], []
    async for row in sample_ratings(pool, prob=proportion):
        users.append(row["user_id"])
        movies.append(row["movie_id"])
        ratings.append(row["rating"])

    users_2_ids = {user: i for i, user in enumerate(set(users))}
    movies_2_ids = {user: i for i, user in enumerate(set(movies))}

    dataset = [
        ([value / 10, movies_2_ids[movie], users_2_ids[user]])
        for value, movie, user in zip(ratings, movies, users)
    ]
    return dataset, users_2_ids, movies_2_ids


def train_kmf_model(
    dataset,
    users_2_ids,
    movies_2_ids,
    emb_dim: int = 40,
    test_size: float = 0.2,
    alpha: float = 0.01,
    lr: float = 1e-2,
    epochs: int = 20,
    stop_after: int = 2,
    verbose: bool = False,
):
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = KMF(len(users_2_ids), len(movies_2_ids), emb_dim, max_score=5)
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    test_losses = []
    for epoch in range(epochs):
        mse_acc = l2_acc = steps = 0
        model.train()
        for scores, items, users in train_loader:
            scores = scores.to(device)
            items = items.to(device)
            users = users.to(device)
            optimizer.zero_grad()
            pred, users_emb, items_emb, users_bias, items_bias = model(users, items)
            mse_loss = mse(scores, pred)
            l2_loss = alpha * get_param_squared_norms(
                users_emb, items_emb, users_bias, items_bias
            )
            mse_acc += float(mse_loss)
            l2_acc += float(l2_loss)
            loss = mse_loss + l2_loss
            loss.backward()
            steps += 1
            optimizer.step()
        mse_acc /= steps
        l2_acc /= steps
        if verbose:
            print(f"Epoch {epoch + 1} -> MSE = {mse_acc:.3f} | L2 reg = {l2_acc:.2f}")

        test_loss = test_steps = 0
        model.eval()
        with torch.no_grad():
            for (scores, items, users) in test_loader:
                scores = scores.to(device)
                items = items.to(device)
                users = users.to(device)
                pred, *_ = model(users, items)
                mse_loss = mse(scores, pred)
                test_loss += float(mse_loss)
                test_steps += 1
        test_loss /= test_steps
        test_losses.append(test_loss)
        if verbose:
            print(f"Test: MSE = {test_loss:.3f}")
        if len(test_losses) > stop_after and np.all(np.diff(test_losses)[-stop_after:] > -1e-3):
            break
    return model


async def run_train_pipeline(
    pool: asyncpg.Pool,
    emb_dim: int,
    proportion: float,
    test_split: float,
    alpha: float = 0.01,
    verbose: bool = False,
) -> None:
    """Run a full training and store vectors in the database."""
    dataset, users_2_ids, movies_2_ids = await fetch_data(pool, proportion, test_split)
    model = train_kmf_model(
        dataset, users_2_ids, movies_2_ids, emb_dim, alpha=alpha, verbose=verbose
    )

    await delete_all_movie_vector_bias(pool)
    await delete_all_user_vector_bias(pool)

    await write_model_data(
        model.user_emb,
        model.user_bias,
        users_2_ids,
        write_bulk_user_vector_bias,
        pool
    )

    await write_model_data(
        model.item_emb,
        model.item_bias,
        movies_2_ids,
        write_bulk_movie_vector_bias,
        pool,
    )
