#!/bin/bash

MODEL=large-v2
PROJECT_NAME=whisperx-api
VERSION=v0.1.0-$MODEL

IMAGE_NAME=${PROJECT_NAME}:${VERSION}

docker login
docker build --platform="linux/amd64" --build-arg MODEL=$MODEL -t $IMAGE_NAME .
docker tag $IMAGE_NAME vonsovsky/$IMAGE_NAME
docker push vonsovsky/$IMAGE_NAME
