#!/bin/bash
IMAGE_NAME=country_recognition
CONTAINER_NAME=country_recognition-container
PROJECT_ROOT=$(dirname $(pwd))

docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# start new container with volume
if docker run -d \
              --name $CONTAINER_NAME \
              -v $PROJECT_ROOT:/app \
              -w /app \
              $IMAGE_NAME \
              tail -f /dev/null; then
    echo "🚀 container $CONTAINER_NAME is running"
else
    echo "❌ run failed"
    exit 1
fi
