CUDA_VISIBLE_DEVICES=0,1,2,3
DATASETS="/shared/nas/data/m1/wangz3/Shared_Datasets/VL" 
# from ruochen
docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$DATASETS,dst=/src/shared_datasets,type=bind,readonly \
    -w /src ruox/blip_environment_a100
    # --user "$(id -u):$(id -g)" \
    # --mount src=$IMAGE_ROOT,dst=/src/dataset/COCO/images,type=bind,readonly \


# docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --network=host --rm -it \
#     --mount src=$(pwd),dst=/src,type=bind \
#     --mount src=$IMAGE_ROOT,dst=/src/dataset/COCO/images,type=bind,readonly \
#     -w /src mikewangwzhl/blip_environment_a100