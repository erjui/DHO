import argparse
import os
import random

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import clip
from datasets.imagenet_percent import ImageNetPercent


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--percent', type=float, default=1.0,
                       help='Percentage of labeled data to use (default: 1.0)')
    parser.add_argument('--teacher_model', type=str, default='ViT-L-14',
                        choices=['ViT-L-14', 'ViT-L-14-336px', 'apple/DFN5B-CLIP-ViT-H-14-378'],
                        help='Teacher model architecture to use')
    parser.add_argument('--student_model', type=str, default='ViT-B-16',
                       choices=['RN50', 'ViT-B-16', 'ViT-L-14', 'ViT-L-14-336px'],
                       help='Student model architecture to use')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--train_epoch', type=int, default=32,
                        help='Number of training epochs')
    parser.add_argument('--root_path', type=str, default='./data',
                       help='Root path for dataset')
    args = parser.parse_args()

    return args


def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()

            # Prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


class DualSizeDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, student_size, teacher_size, is_train=True):
        self.base_dataset = base_dataset

        # Define shared augmentations that should be identical
        if is_train:
            self.shared_transforms = transforms.Compose([
                transforms.RandomResizedCrop(teacher_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.shared_transforms = transforms.Compose([
                transforms.Resize(teacher_size),
                transforms.CenterCrop(teacher_size),
            ])

        # Define size-specific transforms
        self.student_resize = transforms.Resize(
            student_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.teacher_resize = transforms.Resize(
            teacher_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Apply shared augmentations first
        augmented_image = self.shared_transforms(image)

        # Then resize separately for student and teacher
        student_image = self.to_tensor(self.student_resize(augmented_image))
        teacher_image = self.to_tensor(self.teacher_resize(augmented_image))

        return student_image, teacher_image, label

    def __len__(self):
        return len(self.base_dataset)


class StudentModel(nn.Module):
    def __init__(self, num_classes, imagenet=None, model_name='ViT-B-16'):
        super().__init__()
        # Use CLIP's vision transformer as backbone
        self.backbone, _ = clip.load(f'./ckpt/clip/{model_name}.pt')

        with torch.no_grad():
            clip_weights = []

            for classname in imagenet.classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in imagenet.template]
                texts = clip.tokenize(texts).cuda()

                # prompt ensemble for ImageNet
                class_embeddings = self.backbone.encode_text(texts)
                class_embedding = class_embeddings.mean(dim=0)
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).cuda()

        self.backbone = self.backbone.float()
        self.backbone = self.backbone.visual

        # Set in_features based on model architecture
        in_features = {
            'RN50': 1024,
            'ViT-B-16': 512,
            'ViT-L-14': 768,
            'ViT-L-14-336px': 768
        }[model_name]

        # Add two branches
        self.ce_head = nn.Linear(in_features, num_classes)  # CE branch
        self.kd_head = nn.Linear(in_features, num_classes)  # KD branch

        # Initialize CLIP text weights and zero bias
        with torch.no_grad():
            self.ce_head.weight.copy_(clip_weights.float().T)
            self.kd_head.weight.copy_(clip_weights.float().T)
            self.ce_head.bias.zero_()
            self.kd_head.bias.zero_()

    def forward(self, x):
        features = self.backbone(x)
        ce_out = self.ce_head(features)
        kd_out = self.kd_head(F.normalize(features, dim=1)) * 100
        return ce_out, kd_out


def train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                 train_loader_labeled, train_loader_unlabeled, test_loader,
                 clip_weights):

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Setup training
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5000),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.train_epoch * len(train_loader_unlabeled) - 5000)
    ])

    # Initialize training parameters
    temperature = 2.0
    best_acc, best_epoch, best_alpha, best_beta = 0.0, 0, 0.5, 1.0
    cache_path = os.path.join('./logs', args.dataset, f"ckpt_dho_t-{args.teacher_model}_s-{args.student_model}_p-{args.percent}.pt")

    for epoch in range(args.train_epoch):
        student_model.train()
        total_loss = 0

        # Set samplers' epoch for proper shuffling
        train_loader_labeled.sampler.set_epoch(epoch)
        train_loader_unlabeled.sampler.set_epoch(epoch)

        if rank == 0:
            print(f'\n📊 Training Epoch: {epoch+1}/{args.train_epoch}')

        labeled_iterator = iter(train_loader_labeled)
        pbar = tqdm(train_loader_unlabeled, desc='Training',
                   disable=rank != 0)

        for unlabeled_imgs_student, unlabeled_imgs_clip, _ in pbar:
            try:
                labeled_imgs_student, labeled_imgs_clip, labels = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(train_loader_labeled)
                labeled_imgs_student, labeled_imgs_clip, labels = next(labeled_iterator)

            # Move to GPU and normalize
            labeled_imgs_student = student_normalize(labeled_imgs_student.cuda())
            labeled_imgs_clip = clip_normalize(labeled_imgs_clip.cuda())
            unlabeled_imgs_student = student_normalize(unlabeled_imgs_student.cuda())
            unlabeled_imgs_clip = clip_normalize(unlabeled_imgs_clip.cuda())
            labels = labels.cuda()

            # Get teacher predictions
            with torch.no_grad():
                # Handle fp16 conversion if needed
                if list(clip_model.parameters())[1].dtype == torch.float16:
                    labeled_imgs_clip = labeled_imgs_clip.half()
                    unlabeled_imgs_clip = unlabeled_imgs_clip.half()

                # Get teacher predictions for labeled data
                clip_features = clip_model.encode_image(labeled_imgs_clip)
                if clip_features.dtype == torch.float16:
                    clip_features = clip_features.float()
                clip_features /= clip_features.norm(dim=-1, keepdim=True)
                teacher_logits_labeled = 100. * clip_features @ clip_weights.float()

                # Get teacher predictions for unlabeled data
                clip_features = clip_model.encode_image(unlabeled_imgs_clip)
                if clip_features.dtype == torch.float16:
                    clip_features = clip_features.float()
                clip_features /= clip_features.norm(dim=-1, keepdim=True)
                teacher_logits_unlabeled = 100. * clip_features @ clip_weights.float()

            # Get student predictions
            stacked_imgs = torch.cat([labeled_imgs_student, unlabeled_imgs_student], dim=0)
            stacked_logits_ce, stacked_logits_kd = student_model(stacked_imgs)
            student_logits_labeled_ce = stacked_logits_ce[:labeled_imgs_student.size(0)]
            student_logits_labeled_kd = stacked_logits_kd[:labeled_imgs_student.size(0)]
            _ = stacked_logits_ce[labeled_imgs_student.size(0):]
            student_logits_unlabeled_kd = stacked_logits_kd[labeled_imgs_student.size(0):]

            # Calculate losses
            ce_loss = F.cross_entropy(student_logits_labeled_ce, labels)

            distill_loss_labeled = F.kl_div(
                F.log_softmax(student_logits_labeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_labeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            distill_loss_unlabeled = F.kl_div(
                F.log_softmax(student_logits_unlabeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_unlabeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            loss = ce_loss + (distill_loss_labeled + distill_loss_unlabeled)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_dict = {
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                'CE': f'{ce_loss.item():.4f}',
                'Dist_L': f'{distill_loss_labeled.item():.4f}',
                'Dist_U': f'{distill_loss_unlabeled.item():.4f}'
            }
            pbar.set_postfix(progress_dict)
            total_loss += loss.item()

        # Gather metrics from all processes
        total_loss = torch.tensor(total_loss).cuda()
        dist.all_reduce(total_loss)
        total_loss = total_loss.item()

        # Evaluate with parameter search
        test_acc, alpha, beta = evaluate(student_model, test_loader, student_normalize)

        # Print epoch results only on rank 0
        if rank == 0:
            avg_loss = total_loss / len(train_loader_unlabeled)
            print(f'Epoch: {epoch+1}/{args.train_epoch}')
            print(f'Loss: {avg_loss:.4f}')
            print(f'Test Acc: {test_acc:.2f}%')
            print(f'Best parameters: alpha={alpha:.2f}, beta={beta:.2f}\n')

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best_alpha = alpha
                best_beta = beta
                torch.save({
                    'model_state_dict': student_model.state_dict(),
                    'epoch': epoch,
                    'acc': test_acc,
                    'alpha': alpha,
                    'beta': beta
                }, cache_path)

    # Final results only on rank 0
    if rank == 0:
        print(f"\nBest test accuracy: {best_acc:.2f}% at epoch {best_epoch}")
        print(f"Best parameters: alpha={best_alpha:.2f}, beta={best_beta:.2f}")


def evaluate(model, data_loader, normalize, alpha=0.5, beta=0.5):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for (images_student, _, labels) in tqdm(data_loader, desc='Evaluating',
                                 disable=dist.get_rank() if dist.is_initialized() else False):
            images_student, labels = images_student.cuda(), labels.cuda()
            images_student = normalize(images_student)
            outputs_ce, outputs_kd = model(images_student)

            probs_ce = F.softmax(outputs_ce, dim=1)
            probs_kd = F.softmax(outputs_kd / beta, dim=1)
            probs = alpha * probs_ce + (1 - alpha) * probs_kd

            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Gather metrics from all processes
    correct = torch.tensor(correct).cuda()
    total = torch.tensor(total).cuda()
    dist.all_reduce(correct)
    dist.all_reduce(total)
    correct = correct.item()
    total = total.item()

    acc = 100 * correct / total
    return acc, alpha, beta


def main():
    # Load arguments
    args = get_arguments()
    process_count = int(os.environ.get('WORLD_SIZE', 1))
    args.batch_size = args.batch_size // process_count
    args.local_rank = int(os.environ.get('RANK', -1))
    print(f'Using distributed training with {process_count} processes. Batch size per process: {args.batch_size}')

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    # Set up distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)  # Set GPU device
        dist.init_process_group(backend='nccl')

    # Load the teacher model: use OpenCLIP for 'apple/DFN5B-CLIP-ViT-H-14-378', otherwise fall back to CLIP
    if args.teacher_model == 'apple/DFN5B-CLIP-ViT-H-14-378':
        clip_model, _, _ = open_clip.create_model_and_transforms(
            'ViT-H-14-378-quickgelu',
            pretrained='dfn5b',
            device='cuda',
            precision='fp16'
        )
        teacher_size = 378
    else:
        clip_model, _ = clip.load(f'./ckpt/clip/{args.teacher_model}.pt')
        teacher_size = 336 if args.teacher_model == 'clip-vit-l-14-336' else 224

    clip_model = clip_model.cuda()
    clip_model.eval()
    clip_model.requires_grad_(False)

    # Prepare ImageNet dataset
    print("Preparing ImageNet dataset.")
    imagenet = ImageNetPercent(args.root_path, None, None, percent=args.percent)

    # Update clip_weights calculation for OpenCLIP
    if args.teacher_model == 'apple/DFN5B-CLIP-ViT-H-14-378':
        tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
        text_features = []
        for classname in tqdm(imagenet.classnames, desc='Processing class names'):
            # Process each template
            class_features = []
            for template in imagenet.template:
                text = template.format(classname)
                tokens = tokenizer(text).cuda()
                features = clip_model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
                class_features.append(features)
            # Average features across templates
            class_features = torch.stack(class_features).mean(0)
            text_features.append(class_features)
        clip_weights = torch.stack(text_features).squeeze().T
    else:
        # Original clip_weights calculation
        clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)

    # Determine image sizes based on model architectures
    student_size = 336 if args.student_model == 'clip-vit-l-14-336' else 224

    # Setup normalizations (same for both since they're both CLIP models)
    student_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    clip_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # Wrap datasets with dual transforms
    train_dataset_labeled = DualSizeDataset(imagenet.train_x, student_size, teacher_size)
    train_dataset_unlabeled = DualSizeDataset(imagenet.train_u, student_size, teacher_size)
    test_dataset = DualSizeDataset(imagenet.test, student_size, teacher_size)


    # Update dataloaders
    train_sampler_labeled = DistributedSampler(train_dataset_labeled)
    train_sampler_unlabeled = DistributedSampler(train_dataset_unlabeled)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader_labeled = torch.utils.data.DataLoader(
        train_dataset_labeled, batch_size=args.batch_size,
        sampler=train_sampler_labeled,
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )
    train_loader_unlabeled = torch.utils.data.DataLoader(
        train_dataset_unlabeled, batch_size=args.batch_size,
        sampler=train_sampler_unlabeled,
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )

    # Initialize student model with selected architecture
    student_model = StudentModel(
        num_classes=len(imagenet.classnames),
        imagenet=imagenet,
        model_name=args.student_model
    ).cuda()
    student_model = DDP(student_model, device_ids=[args.local_rank])

    # Since we're using the same architecture, we can use the same normalization
    student_normalize = clip_normalize  # Use CLIP's normalization for both

    # Train student model
    train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                 train_loader_labeled, train_loader_unlabeled, test_loader,
                 clip_weights)


if __name__ == '__main__':
    main()
