#!/bin/bash -eu
#set -eu
project_root="$(cd "$(dirname "${BASH_SOURCE:-$0}")"/../../; pwd)"
ssd_root=${project_root}/SSD-pytorch
image_tag=ssdres50
cd "${ssd_root}"
DOCKER_BUILDKIT=1 docker build --tag "${image_tag}" -f Dockerfile --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" .
