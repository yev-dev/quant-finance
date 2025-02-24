#!/bin/bash

export VERSION=0.1
export IMAGE_NAME=qf
export DOCKER_USER_HOME=/home/qf-user
export PORT_NUM=9999

echo "Running ${IMAGE_NAME} image for ${VERSION} version tag"

docker run --name ${IMAGE_NAME} -it --rm --detach -p {PORT_NUM}
--volume ${HOME}/notebooks/qf:${DOCKER_USER HOME}/notebooks
--volume ${HOME}/data/qf:${DOCKER_USER_HOME}/notebooks/data
${IMAGE}:${VERSION}
