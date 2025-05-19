#!/bin/bash

# Define arrays for iterations
DATASETS=(
    "caltech101"
    "dtd"
    "eurosat"
    "fgvc_aircraft"
    "food101"
    "oxford_flowers"
    "oxford_pets"
    "stanford_cars"
    "sun397"
    "ucf101"
)
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
STUDENT_MODELS=(
    "res18"
    "mobilenet"
)

# Loop over all combinations
for dataset in "${DATASETS[@]}"; do
    for shots in "${SHOTS[@]}"; do
        for teacher_type in "${TEACHER_TYPES[@]}"; do
            for student_model in "${STUDENT_MODELS[@]}"; do

                echo "Running: dataset=$dataset, shots=$shots, teacher_type=$teacher_type, student_model=$student_model"

                if [ "$teacher_type" = "zs" ]; then
                    # Zero-shot teacher command
                    CUDA_VISIBLE_DEVICES=0 python train_others.py \
                        --dataset $dataset \
                        --shots $shots \
                        --teacher_type zs \
                        --student_model $student_model \
                        --batch_size 64 \
                        --train_epoch 200 \
                        --lr 0.001 \
                        --root_path ./data \
                        | tee ./logs/$dataset/dho_${student_model}_zs_${shots}shot.log
                elif [ "$teacher_type" = "fs" ]; then
                    # Few-shot teacher command
                    CUDA_VISIBLE_DEVICES=1 python train_others.py \
                        --dataset $dataset \
                        --shots $shots \
                        --teacher_type fs \
                        --teacher_ckpt ./ckpt/fewshot_teacher/$dataset/tip_adapter_${shots}shot/best_tip_adapter_F_${shots}shots_round0.pt \
                        --student_model $student_model \
                        --batch_size 64 \
                        --train_epoch 200 \
                        --lr 0.001 \
                        --root_path ./data \
                        | tee ./logs/$dataset/dho_${student_model}_fs_${shots}shot.log
                fi

                # Add a small delay between runs to avoid potential issues
                sleep 5
            done
        done
    done
done
