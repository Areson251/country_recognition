#!/bin/bash
IMAGE_NAME=country_recognition

cd ..

if docker build -t $IMAGE_NAME -f docker/Dockerfile .; then
    echo "✅ done $IMAGE_NAME building"
else
    echo "❌ build failed"
    exit 1
fi
