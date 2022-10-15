import asyncio
import random
import asyncpg

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.db.postgres import get_all_ratings, write_user_vector_bias, write_movie_vector_bias, sample_ratings
from app.ml.kmf import KMF, get_param_squared_norms, mse
from app.models import VectorBias


POSTGRES_URI = "postgresql://postgres:postgres@localhost:5401/movies"


async def main():
    random.seed(42)
    torch.random.manual_seed(42)
    pool = await asyncpg.create_pool(POSTGRES_URI)

    users = []
    movies = []
    ratings = []
    #async for row in sample_ratings(pool):
    async for row in get_all_ratings(pool):
        users.append(row['user_id'])
        movies.append(row['movie_id'])
        ratings.append(row['rating'])

    users_2_ids = {user: i for i, user in enumerate(set(users))}
    movies_2_ids = {user: i for i, user in enumerate(set(movies))}
    ids_2_users = {i: user for user, i in users_2_ids.items()}
    ids_2_movie = {i: movie for movie, i in movies_2_ids.items()}


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = [
        (torch.tensor([value / 10, movies_2_ids[movie], users_2_ids[user]]).float().to(device))
        for value, movie, user in zip(ratings, movies, users)
    ]
    random.shuffle(dataset)

    split_point = int(0.8 * len(dataset))
    train_dataset = dataset[:split_point]
    test_dataset = dataset[split_point:]
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    model = KMF(len(users_2_ids), len(movies_2_ids), 40, 5)
    model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    alpha = 0.1

    test_losses = []
    for epoch in range(20):
        mse_acc = l2_acc = steps = 0
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            scores = batch[:, 0]
            items = batch[:, 1].long()
            users = batch[:, 2].long()
            pred, users_emb, items_emb, users_bias, items_bias = model(users, items)
            mse_loss = mse(scores, pred)
            l2_loss = alpha * get_param_squared_norms(users_emb, items_emb, users_bias, items_bias)
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
            for batch in test_loader:
                scores = batch[:, 0]
                items = batch[:, 1].long()
                users = batch[:, 2].long()
                pred, *_ = model(users, items)
                mse_loss = mse(scores, pred)
                test_loss += float(mse_loss)
                test_steps += 1
        test_loss /= test_steps
        test_losses.append(test_loss)
        print(f"Test: MSE = {test_loss:.3f}")
        if len(test_losses) > 3 and np.all(np.diff(test_losses)[-3:] > -1e-3):
            print("stopping training")
            break

    for user, i in users_2_ids.items():
        vector = model.user_emb.weight[i, :].detach().tolist()
        bias = model.user_bias[i].detach().item()
        vector_bias = VectorBias(vector, bias)
        await write_user_vector_bias(pool, vector_bias, user)

    for movie, i in movies_2_ids.items():
        vector = model.item_emb.weight[i, :].detach().tolist()
        bias = model.item_bias[i].detach().item()
        vector_bias = VectorBias(vector, bias)
        await write_movie_vector_bias(pool, vector_bias, movie)


if __name__ == '__main__':
    asyncio.run(main())
