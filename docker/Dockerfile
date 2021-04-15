FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04

ARG APP_DIR=/app

RUN apt-get update && apt-get -y install python3 python3-pip &&  ln /usr/bin/python3 /usr/bin/python && \
    pip3 install --user --upgrade pip && python -m pip install dvc[s3]

WORKDIR $APP_DIR
COPY requirements.txt $APP_DIR/requirements.txt

RUN python -m pip install -r requirements.txt
