import argparse
import asyncio
import random
import asyncpg

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from app.db.postgres import (
    delete_all_movie_vector_bias,
    delete_all_user_vector_bias,
    write_bulk_movie_vector_bias,
    write_bulk_user_vector_bias,
    sample_ratings,
)
from app.ml.kmf import KMF, get_param_squared_norms, mse
from app.models import VectorBias


POSTGRES_URI = "postgresql://postgres:postgres@localhost:5401/movies"


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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--proportion",
        type=float,
        default=0.1,
        help="proportion of rating data to use for training",
    )
    parser.add_argument("--test-split", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(42)
    torch.random.manual_seed(42)
    pool = await asyncpg.create_pool(POSTGRES_URI)

    users = []
    movies = []
    ratings = []
    print("fetching data from Postgres")
    async for row in sample_ratings(pool, prob=args.proportion):
        # async for row in get_all_ratings(pool):
        users.append(row["user_id"])
        movies.append(row["movie_id"])
        ratings.append(row["rating"])

    print(f"creating dataset with {len(users)} rows")
    users_2_ids = {user: i for i, user in enumerate(set(users))}
    movies_2_ids = {user: i for i, user in enumerate(set(movies))}
    print(f"unique users = {len(users_2_ids)} | unique movies = {len(movies_2_ids)}")

    print("creating model")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = [
        ([value / 10, movies_2_ids[movie], users_2_ids[user]])
        for value, movie, user in zip(ratings, movies, users)
    ]
    print("craeted dataset")

    train_dataset, test_dataset = train_test_split(dataset, test_size=args.test_split)
    print("craeted train/test")
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    model = KMF(len(users_2_ids), len(movies_2_ids), 40, 5)
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)

    print("starting training")
    test_losses = []
    for epoch in range(20):
        mse_acc = l2_acc = steps = 0
        model.train()
        for i, (scores, items, users) in enumerate(train_loader):
            scores = scores.to(device)
            items = items.to(device)
            users = users.to(device)
            optimizer.zero_grad()
            pred, users_emb, items_emb, users_bias, items_bias = model(users, items)
            mse_loss = mse(scores, pred)
            l2_loss = args.alpha * get_param_squared_norms(
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
        print(f"Test: MSE = {test_loss:.3f}")
        if len(test_losses) > 2 and np.all(np.diff(test_losses)[-2:] > -1e-3):
            print("stopping training")
            break

    print("deleting previous vector data")
    await delete_all_movie_vector_bias(pool)
    await delete_all_user_vector_bias(pool)

    print("inserting new vectors")
    await write_model_data(
        model.user_emb,
        model.user_bias,
        users_2_ids,
        write_bulk_user_vector_bias,
        pool,
    )
    await write_model_data(
        model.item_emb,
        model.item_bias,
        movies_2_ids,
        write_bulk_movie_vector_bias,
        pool,
    )


if __name__ == "__main__":
    asyncio.run(main())
