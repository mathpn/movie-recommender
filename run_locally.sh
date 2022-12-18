#!/bin/bash
export PYTHONPATH=.:app
export POSTGRES_URI='postgresql://postgres:postgres@localhost:5401/movies'
export DEBUG=1
uvicorn main:app --reload