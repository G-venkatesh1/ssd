#!/bin/bash -eu
project_root="$(cd "$(dirname "${BASH_SOURCE:-$0}")"/../../; pwd)"
image_tag=ssdres50
container_name=ssdres50
data_root=$DATA_DIR
docker run --gpus all --rm -it --shm-size=256m -v "${project_root}"/SSD-pytorch:/workspace/SSD-pytorch \
-v "$data_root:/workspace/SSD-pytorch/datasets_coco" --name ${container_name} ${image_tag}
