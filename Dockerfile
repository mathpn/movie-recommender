FROM python:3.10-slim
WORKDIR /home/
ADD requirements.txt requirements.txt
RUN apt-get update && apt-get -y install gcc
RUN pip install -r requirements.txt
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN mkdir app scripts

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000/tcp
