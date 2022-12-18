#!/bin/bash
export PYTHONPATH=.:app
docker run --name postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=movies -p 5401:5432 -d postgres
python scripts/insert_data_into_db.py