#!/bin/bash

PERCENTS=(
    1.0
    10.0
)
STUDENT_MODELS=(
    "ViT-B-16"
    "ViT-L-14-336px"
)
TEACHER_MODELS=(
    "apple/DFN5B-CLIP-ViT-H-14-378"
)

# Create log directory if it doesn't exist
mkdir -p ./logs/imagenet

for percent in "${PERCENTS[@]}"; do
    for student_model in "${STUDENT_MODELS[@]}"; do
        for teacher_model in "${TEACHER_MODELS[@]}"; do
            echo "Running: percent=$percent, student_model=$student_model, teacher_model=$teacher_model"

            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 train_imgnet_semi.py \
                --teacher_model "$teacher_model" \
                --student_model "$student_model" \
                --lr 5e-5 \
                --train_epoch 32 \
                --batch_size 256 \
                --percent $percent \
                | tee ./logs/imagenet/dho_student_${safe_student_model}_teacher_${safe_teacher_model}_${percent}percent_ep32_lr5e-5.log

            sleep 5
        done
    done
done
