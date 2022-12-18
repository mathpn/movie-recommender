import asyncio
import os

import asyncpg

from app.db.postgres import (create_global_table, create_movies_table,
                             create_ratings_table, create_users_table)
from app.logger import logger


async def main():
    pool = await asyncpg.create_pool(os.environ["POSTGRES_URI"])

    logger.info("creating database tables")
    await create_users_table(pool)
    await create_global_table(pool)
    await create_movies_table(pool)
    await create_ratings_table(pool)
    logger.info("created all required tables")


if __name__ == "__main__":
    asyncio.run(main())
