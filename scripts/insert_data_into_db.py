"""
Insert all relevant data into Database.
"""
import json
from ast import literal_eval

import pandas as pd
import numpy as np
import asyncpg


def get_genres(genres):
    return [x["name"] for x in genres]


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


def is_integer(field):
    try:
        int(field)
        return True
    except Exception:
        return False


def main():
    movies_metadata = pd.read_csv("./data/movies_metadata.csv")
    movie_credits = pd.read_csv("./data/credits.csv")
    keywords = pd.read_csv("./data/keywords.csv")

    movies_metadata = movies_metadata[movies_metadata["id"].apply(is_integer)]
    movies_metadata = movies_metadata.astype({"id": "int64"})
    movies_metadata = movies_metadata.merge(movie_credits, on="id")
    movies_metadata = movies_metadata.merge(keywords, on="id")

    features = ["cast", "crew", "keywords", "genres"]
    for feature in features:
        movies_metadata[feature] = movies_metadata[feature].apply(literal_eval)

    print(len(set(x for sublist in movies_metadata["genres"].apply(get_genres) for x in sublist)))

    movies_metadata["director"] = movies_metadata["crew"].apply(get_director)

    features = ["cast", "keywords", "genres"]
    for feature in features:
        movies_metadata[feature] = movies_metadata[feature].apply(get_list)
    # TODO continue


if __name__ == "__main__":
    main()
