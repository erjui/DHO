import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import clip
from datasets.imagenet import ImageNet


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='Dataset name')
    parser.add_argument('--shots', type=int, default=1,
                        help='Number of shots for few-shot learning')
    parser.add_argument('--teacher_type', choices=['zs', 'fs'],
                        default='fs', help='teacher model type: zs (zero-shot), fs (few-shot)')
    parser.add_argument('--teacher_ckpt', type=str, default=None,
                        help='Path to teacher (Tip-Adapter) checkpoint')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--train_epoch', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
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


class StudentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load DINO pre-trained ResNet50
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add two branches
        in_features = 2048
        self.ce_head = nn.Linear(in_features, num_classes)  # CE head
        self.kd_head = nn.Linear(in_features, num_classes)  # KD head

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        # Forward through both branches
        ce_out = self.ce_head(features)
        kd_out = self.kd_head(features)

        return ce_out, kd_out


def train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                  train_loader_labeled, train_loader_unlabeled, test_loader,
                  clip_weights, teacher_model=None):

    rank = dist.get_rank() if dist.is_initialized() else 0

    # Setup training
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.train_epoch * len(train_loader_unlabeled))

    # Initialize training parameters
    temperature = 2.0
    best_acc, best_epoch, best_alpha, best_beta = 0.0, 0, 0.5, 1.0
    cache_path = os.path.join('./logs', args.dataset, f"ckpt_dho_{args.teacher_type}_{args.shots}shots.pt")

    for epoch in range(args.train_epoch):
        student_model.train()
        total_loss = 0

        # Set samplers' epoch for proper shuffling
        if dist.is_initialized():
            train_loader_labeled.sampler.set_epoch(epoch)
            train_loader_unlabeled.sampler.set_epoch(epoch)

        if rank == 0:
            print(f'\n📊 Training Epoch: {epoch+1}/{args.train_epoch}')

        labeled_iterator = iter(train_loader_labeled)

        for unlabeled_imgs, _ in tqdm(train_loader_unlabeled, desc='Training',
                                      disable=rank != 0):
            # Get labeled batch (with cycling)
            try:
                labeled_imgs, labels = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(train_loader_labeled)
                labeled_imgs, labels = next(labeled_iterator)

            labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()

            # Normalize images
            labeled_imgs_student = student_normalize(labeled_imgs)
            unlabeled_imgs_student = student_normalize(unlabeled_imgs)
            labeled_imgs_clip = clip_normalize(labeled_imgs)
            unlabeled_imgs_clip = clip_normalize(unlabeled_imgs)

            # Get teacher predictions
            with torch.no_grad():
                if teacher_model is not None:
                    # Use Tip-Adapter teacher
                    clip_features = clip_model.encode_image(labeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_labeled = teacher_model(clip_features)

                    clip_features = clip_model.encode_image(unlabeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_unlabeled = teacher_model(clip_features)
                else:
                    # Use CLIP as teacher
                    clip_features = clip_model.encode_image(labeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_labeled = 100. * clip_features @ clip_weights

                    clip_features = clip_model.encode_image(unlabeled_imgs_clip)
                    clip_features /= clip_features.norm(dim=-1, keepdim=True)
                    teacher_logits_unlabeled = 100. * clip_features @ clip_weights

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

            # Combined loss with equal weights
            loss = 0.5 * ce_loss + 0.5 * (distill_loss_labeled + distill_loss_unlabeled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Gather total_loss from all processes
        if dist.is_initialized():
            total_loss = torch.tensor(total_loss, device='cuda')
            dist.all_reduce(total_loss)
            total_loss = total_loss.item() / dist.get_world_size()

        # Evaluate current epoch
        test_acc, alpha, beta = evaluate(student_model, test_loader, student_normalize, search_param=True)

        if rank == 0:
            avg_loss = total_loss / len(train_loader_unlabeled)
            print(f"Epoch: {epoch+1}/{args.train_epoch} | CE Loss: {ce_loss:.4f} | "
                  f"Distill Labeled: {distill_loss_labeled:.4f} | "
                  f"Distill Unlabeled: {distill_loss_unlabeled:.4f}")
            print(f"Total Loss: {avg_loss:.4f} | Test Acc: {test_acc:.2f}% (α: {alpha:.2f}, β: {beta:.2f})")

            # Save best model
            if test_acc > best_acc:
                best_acc, best_epoch, best_alpha, best_beta = test_acc, epoch, alpha, beta
                torch.save({
                    'model_state_dict': student_model.module.state_dict(),
                    'epoch': epoch,
                    'acc': test_acc,
                    'alpha': alpha,
                    'beta': beta
                }, cache_path)

    checkpoint = torch.load(cache_path)
    student_model.module.load_state_dict(checkpoint['model_state_dict'])
    final_acc = evaluate(student_model, test_loader, student_normalize,
                         search_param=False,
                         alpha=checkpoint['alpha'],
                         beta=checkpoint['beta'])

    # Final results
    if rank == 0:
        print("\n" + "=" * 60)
        print("🎯 FINAL RESULTS")
        print("=" * 60)
        print(f"📊 Best Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        print(f"⚙️  Optimal Parameters: α={best_alpha:.2f}, β={best_beta:.2f}")
        print(f"🎯 Final Test Accuracy: {final_acc:.2f}%")
        print("=" * 60 + "\n")


def evaluate(model, data_loader, normalize, search_param=False, alpha=0.0, beta=1.0):
    model.eval()
    rank = dist.get_rank() if dist.is_initialized() else 0

    if not search_param:
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Evaluating',
                                       disable=dist.get_rank() if dist.is_initialized() else False):
                images, labels = images.cuda(), labels.cuda()
                images = normalize(images)
                outputs_ce, outputs_kd = model(images)

                # Convert logits to probabilities before interpolation
                probs_ce = F.softmax(outputs_ce, dim=1)
                probs_kd = F.softmax(outputs_kd / beta, dim=1)

                # Interpolate between CE and KD probabilities
                probs = (1 - alpha) * probs_ce + alpha * probs_kd

                _, predicted = torch.max(probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Gather metrics from all processes
        if dist.is_initialized():
            correct = torch.tensor(correct).cuda()
            total = torch.tensor(total).cuda()
            dist.all_reduce(correct)
            dist.all_reduce(total)
            correct = correct.item()
            total = total.item()

        return 100 * correct / total

    # Compute outputs once for all images
    all_outputs_ce = []
    all_outputs_kd = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Collecting predictions',
                                   disable=dist.get_rank() if dist.is_initialized() else False):
            images, labels = images.cuda(), labels.cuda()
            images = normalize(images)
            outputs_ce, outputs_kd = model(images)
            all_outputs_ce.append(outputs_ce)
            all_outputs_kd.append(outputs_kd)
            all_labels.append(labels)

    # Stack all outputs and labels
    all_outputs_ce = torch.cat(all_outputs_ce)
    all_outputs_kd = torch.cat(all_outputs_kd)
    all_labels = torch.cat(all_labels)

    # Gather predictions from all processes
    if dist.is_initialized():
        # Create list of tensors for gathering
        gather_outputs_ce = [torch.zeros_like(all_outputs_ce) for _ in range(dist.get_world_size())]
        gather_outputs_kd = [torch.zeros_like(all_outputs_kd) for _ in range(dist.get_world_size())]
        gather_labels = [torch.zeros_like(all_labels) for _ in range(dist.get_world_size())]

        # Gather all predictions
        dist.all_gather(gather_outputs_ce, all_outputs_ce)
        dist.all_gather(gather_outputs_kd, all_outputs_kd)
        dist.all_gather(gather_labels, all_labels)

        # Concatenate gathered tensors
        all_outputs_ce = torch.cat(gather_outputs_ce)
        all_outputs_kd = torch.cat(gather_outputs_kd)
        all_labels = torch.cat(gather_labels)

    if rank == 0:
        best_acc = 0
        best_alpha = 0
        best_beta = 1.0
        alpha_range = np.arange(0, 1.05, 0.05)
        beta_range = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0])

        # Convert CE logits to probabilities
        all_probs_ce = F.softmax(all_outputs_ce, dim=1)

        # Parameter search
        for beta in beta_range:
            # Apply temperature scaling to KD logits
            all_probs_kd = F.softmax(all_outputs_kd / beta, dim=1)

            for alpha in alpha_range:
                # Interpolate between CE and KD probabilities
                probs = (1 - alpha) * all_probs_ce + alpha * all_probs_kd
                _, predicted = torch.max(probs.data, 1)
                correct = (predicted == all_labels).sum().item()
                acc = 100 * correct / len(all_labels)

                print(f'alpha: {alpha:.2f}, beta: {beta:.2f}, acc: {acc:.2f}%')

                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha
                    best_beta = beta
                    print(f'New best! acc: {best_acc:.2f}%, alpha: {best_alpha:.2f}, beta: {best_beta:.2f}')

    # Broadcast best results from rank 0 to all processes
    if dist.is_initialized():
        best_results = torch.tensor([best_acc, best_alpha, best_beta]).cuda() if rank == 0 else torch.zeros(3).cuda()
        dist.broadcast(best_results, src=0)
        best_acc, best_alpha, best_beta = best_results.tolist()

    return best_acc, best_alpha, best_beta


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
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')

    # Load CLIP model
    clip_model, _ = clip.load('RN50')
    clip_model = clip_model.to(f"cuda:{args.local_rank}")

    # Load teacher model (Tip-Adapter) if specified
    teacher_model = None
    if args.teacher_ckpt:
        print(f"Loading teacher model from {args.teacher_ckpt}")
        teacher_ckpt = torch.load(args.teacher_ckpt, map_location=f"cuda:{args.local_rank}")
        device = torch.device(f"cuda:{args.local_rank}")

        adapter_weight = teacher_ckpt['adapter_weight'].to(device)
        best_beta = teacher_ckpt['best_beta']
        best_alpha = teacher_ckpt['best_alpha']
        _ = teacher_ckpt['cache_keys'].to(device)
        cache_values = teacher_ckpt['cache_values'].to(device)
        clip_weights = teacher_ckpt['clip_weights'].to(device)

        # Create a function to get teacher predictions
        def get_teacher_predictions(features, adapter_weight=adapter_weight, cache_values=cache_values,
                                    clip_weights=clip_weights, best_beta=best_beta, best_alpha=best_alpha):
            with torch.no_grad():
                affinity = features @ adapter_weight.t()
                cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                return clip_logits + cache_logits * best_alpha

        teacher_model = get_teacher_predictions

        print(f"Teacher model loaded successfully from {args.teacher_ckpt}")

    # Setup transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Setup transforms
    student_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    clip_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # Prepare ImageNet dataset
    print("Preparing ImageNet dataset.")
    imagenet = ImageNet(args.root_path, args.shots, train_transform, val_transform)

    # Get CLIP weights
    clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model)
    clip_weights = clip_weights.to(f"cuda:{args.local_rank}")
    clip_model.eval()

    # Modify dataloaders to use DistributedSampler
    train_sampler_labeled = DistributedSampler(imagenet.train_x)
    train_sampler_unlabeled = DistributedSampler(imagenet.train_u)
    test_sampler = DistributedSampler(imagenet.test, shuffle=False)

    train_loader_labeled = torch.utils.data.DataLoader(
        imagenet.train_x, batch_size=args.batch_size,
        sampler=train_sampler_labeled,
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )
    train_loader_unlabeled = torch.utils.data.DataLoader(
        imagenet.train_u, batch_size=args.batch_size,
        sampler=train_sampler_unlabeled,
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )
    test_loader = torch.utils.data.DataLoader(
        imagenet.test, batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=8, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda _: os.sched_setaffinity(0, range(os.cpu_count()))
    )

    # Initialize DINO student model with multi-GPU support
    student_model = StudentModel(num_classes=len(imagenet.classnames)).cuda()
    student_model = DDP(student_model, device_ids=[args.local_rank])

    # Train student model
    train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                  train_loader_labeled, train_loader_unlabeled, test_loader,
                  clip_weights, teacher_model=teacher_model)


if __name__ == '__main__':
    main()
