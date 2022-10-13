#!/bin/bash

# Build docker image and push it to the repository

docker build -t jlouradour_wav2vec:latest . || exit 1
docker login registry.linto.ai || exit 1
docker tag jlouradour_wav2vec:latest registry.linto.ai/training/jlouradour_wav2vec:latest || exit 1
docker push registry.linto.ai/training/jlouradour_wav2vec:latest || exit 1
