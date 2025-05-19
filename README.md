# DHO: Simple Few-shot Semi-supervised Knowledge Distillation from Vision-Language Models via Dual-Head Optimization
Official implementation of ['DHO: Simple Few-shot Semi-supervised Knowledge Distillation from Vision-Language Models via Dual-Head Optimization'](https://arxiv.org).

## Requirements
### Installation

```bash
pip install torch torchvision
pip install -r requirements.txt
```

### Dataset

see [DATASET.md](DATASET.md)

### Pretrained teacher models

todo. teacher model checkpoints
todo. pretrained teacher model weights

## Get Started

### Training few-shot semi-supervised distillation

#### ImageNet

zero-shot teacher distillation
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 train_imgnet.py \
    --dataset imagenet \
    --shots 1 \
    --teacher_type zs \
    --batch_size 256 \
    --train_epoch 10 \
    --lr 0.001 \
    | tee ./logs/imagenet/dho_res18_zs_1shot.log
```

few-shot teacher distillation
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 train_imgnet.py \
    --dataset imagenet \
    --shots 1 \
    --teacher_type fs \
    --teacher_ckpt ./ckpt/fewshot_teacher/imagenet/tip_adapter_1shot/best_tip_adapter_F_1shots_round0.pt \
    --batch_size 256 \
    --train_epoch 10 \
    --lr 0.001 \
    | tee ./logs/imagenet/dho_res18_fs_1shot.log
```

#### Other datasets

zero-shot teacher distillation
```bash
CUDA_VISIBLE_DEVICES=0 python train_others.py \
    --dataset caltech101 \
    --shots 1 \
    --teacher_type zs \
    --student_model res18 \
    --batch_size 64 \
    --train_epoch 10 \
    --lr 0.001 \
    --root_path ./data \
    | tee ./logs/caltech101/dho_res18_zs_1shot.log
```

few-shot teacher distillation
```bash
CUDA_VISIBLE_DEVICES=1 python train_others.py \
    --dataset caltech101 \
    --shots 1 \
    --teacher_type fs \
    --teacher_ckpt ./ckpt/fewshot_teacher/caltech101/tip_adapter_1shot/best_tip_adapter_F_1shots_round0.pt \
    --student_model res18 \
    --batch_size 64 \
    --train_epoch 10 \
    --lr 0.001 \
    --root_path ./data \
    | tee ./logs/caltech101/dho_res18_fs_1shot.log
```

### Training VLM-VLM Distillation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 train_imgnet_semi.py \
    --teacher_model ViT-L-14 \
    --student_model ViT-B-16 \
    --lr 5e-5 \
    --train_epoch 32 \
    --batch_size 256 \
    --percent 1.0 \
    | tee ./logs/imagenet/dho_student_vitb_teacher_vitl14_1percent_ep32_lr5e-5.log
```

## Contributors
## Acknowledgement

todo. add tip-adapter code

## Citation
## Contact
