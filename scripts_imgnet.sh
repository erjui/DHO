#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p ./logs/imagenet

SHOTS=(
    1
    2
    4
    8
    16
)
TEACHER_TYPES=(
    "zs"
    "fs"
)

for teacher_type in "${TEACHER_TYPES[@]}"; do
    for shots in "${SHOTS[@]}"; do
        echo "Running: teacher_type=$teacher_type, shots=$shots"
        if [ "$teacher_type" = "zs" ]; then
            CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 train_imgnet.py \
                --dataset imagenet \
                --shots $shots \
                --teacher_type zs \
                --batch_size 256 \
                --train_epoch 20 \
                --lr 0.001 \
                | tee ./logs/imagenet/dho_res18_zs_${shots}shot.log
        elif [ "$teacher_type" = "fs" ]; then
            CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29500 train_imgnet.py \
                --dataset imagenet \
                --shots $shots \
                --teacher_type fs \
                --teacher_ckpt ./ckpt/fewshot_teacher/imagenet/tip_adapter_${shots}shot/best_tip_adapter_F_${shots}shots_round0.pt \
                --batch_size 256 \
                --train_epoch 20 \
                --lr 0.001 \
                | tee ./logs/imagenet/dho_res18_fs_${shots}shot.log
        fi
        sleep 5
    done

done
