#!/bin/bash
ln -fs docker/batchai/.dockerignore.batchai .dockerignore
docker build -t mitcldx.azurecr.io/batchai_worker . -f docker/batchai/Dockerfile
docker push mitcldx.azurecr.io/batchai_worker
