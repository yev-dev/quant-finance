#!/bin/bash

export IMAGE=qf
export VERSIONS=0.1

echo "Building docker file for ${IMAGE} image with ${VERSION} tag"

docker build -t ${IMAGE}:${VERSION} -f docker/Dockerfile .