import asyncio
import re

import asyncpg
from aiocache import cached
from rapidfuzz import distance, process
from unidecode import unidecode

from app.db.postgres import get_all_movie_titles
from app.logger import logger
from app.utils import timed


def _clean_string(string):
    string = unidecode(string).lower()
    return re.sub(r"[^\x00-\x7F]", "", string)


@cached(ttl=600)
@timed
async def get_searcher(pool: asyncpg.Pool):
    movie_ids, movie_titles = await get_all_movie_titles(pool)
    clean_titles = [_clean_string(title) for title in movie_titles]
    ids_2_titles = {idx: title for idx, title in zip(movie_ids, movie_titles)}
    ids_2_clean_titles = {idx: title for idx, title in zip(movie_ids, clean_titles)}

    @timed
    def search(query: str, limit: int = 10) -> list[int]:
        if len(query) == 0:
            raise ValueError("search query is empty")

        query = _clean_string(query)
        # https://maxbachmann.github.io/RapidFuzz/Usage/distance/DamerauLevenshtein.html
        top_matches = process.extract(
            query,
            ids_2_clean_titles,
            limit=limit,
            scorer=distance.JaroWinkler.normalized_distance,
        )
        logger.debug(top_matches)
        return [(movie_id, ids_2_titles[movie_id]) for _, _, movie_id in top_matches]

    return search


async def main():
    pool = await asyncpg.create_pool(
        "postgresql://postgres:postgres@localhost:5401/movies"
    )
    searcher = await get_searcher(pool)
    out = searcher("axé")
    print(out)

    searcher = await get_searcher(pool)
    out = searcher("harry potter")
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
