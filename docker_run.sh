#!/bin/bash

# Build the image
docker build -t conversation:latest . --platform linux/arm64/v8 

# Run the container with environment variables
docker run -p 8501:8501 \
    --env-file .dockerenv \
    conversation:latest