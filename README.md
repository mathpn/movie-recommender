# movie-recommender
A movie recommender built using The Movie Dataset (from Kaggle).

## How it works

# TODO finish this

## The dataset

Please download the data from [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Version 7 was used to develop this project and it's not guarateed to work with future versions. Download the CSV files and save them to the ./data folder inside this project's folder.

Please note that by default the service only uses ratings_small to speed up data import and model training, so the large version of this file are not required. Please note that using the full dataset requires much more memory for model training.


## Developing environment

All dependencies are on requirements.txt. Note that torch is commented out since CPU or GPU version might be wanted and thus it should be installed separately.

### Running with Docker

    docker-compose rm && docker-compose  up --build -d && docker-compose logs --tail 100 -f

The service will be available at http://0.0.0.0:8383/

### Running outside Docker for development and debugging
1. You must have a running local postgres instance and fill it with the data. To do this, please run:

    bash prepare_local.sh

2. Start the service (this requires all the dependencies, including pytorch which is commented out from the requirements)

    bash run_locally.sh

3. The service is available at http://0.0.0.0:8000/