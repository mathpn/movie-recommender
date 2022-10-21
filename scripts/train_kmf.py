"""
Helper script that calls the training pipeline.

Requires a running postgres with data inserted:
    docker run --name postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=movies -p 5401:5432 -d postgres
"""

import argparse
import asyncio
import random

import asyncpg
import torch

from app.ml.kmf import run_train_pipeline

POSTGRES_URI = "postgresql://postgres:postgres@localhost:5401/movies"


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dim", type=int, default=40)
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

    await run_train_pipeline(
        pool,
        args.emb_dim,
        args.proportion,
        args.test_split,
        args.alpha,
        verbose=True
    )


if __name__ == "__main__":
    asyncio.run(main())
