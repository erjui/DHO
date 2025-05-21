import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import transformers
from tqdm import tqdm

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='caltech101',
                        choices=['caltech101', 'dtd', 'eurosat', 'fgvc', 'food101', 'oxford_flowers', 'oxford_pets', 'stanford_cars', 'sun397', 'ucf101'],
                        help='Dataset name')
    parser.add_argument('--shots', type=int, default=1,
                        help='Number of shots (samples per class) for few-shot learning')
    parser.add_argument('--teacher_type', choices=['zs', 'fs'],
                        default='fs', help='teacher model type: zs (zero-shot), fs (few-shot)')
    parser.add_argument('--teacher_ckpt', type=str, default="",
                        help='Path to teacher (Tip-Adapter) checkpoint')
    parser.add_argument('--student_model', choices=['res18', 'mobilenet'], default='res18',
                        help='Student model architecture (ResNet18, or MobileNetV2)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--train_epoch', type=int, default=200,
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
    def __init__(self, num_classes, model_type='res18'):
        super().__init__()
        # Choose backbone based on model type
        if model_type == 'res18':
            self.backbone = models.resnet18(pretrained=True)
            in_features = 512
        elif model_type == 'mobilenet':
            self.backbone = models.mobilenet_v2(pretrained=True)
            in_features = 1280
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Add two branches
        self.ce_head = nn.Linear(in_features, num_classes)
        self.kd_head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        # Forward through both branches
        ce_out = self.ce_head(features)
        kd_out = self.kd_head(features)

        return ce_out, kd_out


def train_student(args, student_model, student_normalize, clip_model, clip_normalize,
                  train_loader_labeled, train_loader_unlabeled, val_loader,
                  test_loader, clip_weights, teacher_model=None):
    # Setup training
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * args.train_epoch * len(train_loader_labeled),
        num_training_steps=args.train_epoch * len(train_loader_labeled)
    )

    # Initialize training parameters
    temperature = 2.0
    best_val_acc, best_epoch, best_alpha, best_beta = 0.0, 0, 0.0, 1.0
    cache_path = os.path.join('./logs', args.dataset, f"ckpt_dho_{args.student_model}_{args.teacher_type}_{args.shots}shots.pt")

    # Main training loop
    for epoch in range(args.train_epoch):
        student_model.train()
        total_loss = 0

        print(f'\n📊 Training Epoch: {epoch+1}/{args.train_epoch}')
        for (labeled_imgs, labels), (unlabeled_imgs, _) in tqdm(
            zip(train_loader_labeled, train_loader_unlabeled),
            total=min(len(train_loader_labeled), len(train_loader_unlabeled)),
            desc='Training batches'
        ):
            # Move data to GPU
            labeled_imgs, labels = labeled_imgs.cuda(), labels.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()

            # Prepare inputs for both student and teacher models
            labeled_imgs_student = student_normalize(labeled_imgs)
            unlabeled_imgs_student = student_normalize(unlabeled_imgs)
            labeled_imgs_clip = clip_normalize(labeled_imgs)
            unlabeled_imgs_clip = clip_normalize(unlabeled_imgs)

            # Generate teacher predictions
            with torch.no_grad():
                # Extract and normalize CLIP features
                labeled_feats = clip_model.encode_image(labeled_imgs_clip)
                labeled_feats = labeled_feats / labeled_feats.norm(dim=-1, keepdim=True)
                unlabeled_feats = clip_model.encode_image(unlabeled_imgs_clip)
                unlabeled_feats = unlabeled_feats / unlabeled_feats.norm(dim=-1, keepdim=True)

                # Generate teacher logits (either from teacher model or CLIP)
                teacher_logits_labeled = teacher_model(labeled_feats) if teacher_model else 100. * labeled_feats @ clip_weights
                teacher_logits_unlabeled = teacher_model(unlabeled_feats) if teacher_model else 100. * unlabeled_feats @ clip_weights

            # Generate student predictions from both heads
            student_logits_labeled_ce, student_logits_labeled_kd = student_model(labeled_imgs_student)
            _, student_logits_unlabeled_kd = student_model(unlabeled_imgs_student)

            # Calculate training losses
            # 1. Standard cross-entropy loss on labeled data
            ce_loss = F.cross_entropy(student_logits_labeled_ce, labels)

            # 2. Knowledge distillation loss on labeled data
            distill_loss_labeled = F.kl_div(
                F.log_softmax(student_logits_labeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_labeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            # 3. Knowledge distillation loss on unlabeled data
            distill_loss_unlabeled = F.kl_div(
                F.log_softmax(student_logits_unlabeled_kd/temperature, dim=1),
                F.softmax(teacher_logits_unlabeled/temperature, dim=1),
                reduction='batchmean'
            ) * (temperature * temperature)

            # Combine losses with equal weights
            loss = 0.5 * ce_loss + 0.5 * (distill_loss_labeled + distill_loss_unlabeled)

            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Evaluate current epoch
        avg_loss = total_loss / len(train_loader_labeled)
        val_acc, alpha, beta = evaluate(student_model, val_loader, student_normalize, search_param=True)

        print(f"Epoch: {epoch+1}/{args.train_epoch} | CE Loss: {ce_loss:.4f} | Distill Labeled: {distill_loss_labeled:.4f} | Distill Unlabeled: {distill_loss_unlabeled:.4f}")
        print(f"Total Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% (α: {alpha:.2f}, β: {beta:.2f})")

        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, best_alpha, best_beta = val_acc, epoch, alpha, beta
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'alpha': best_alpha,
                'beta': best_beta
            }, cache_path)

    # Final evaluation
    checkpoint = torch.load(cache_path)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = evaluate(student_model, test_loader, student_normalize,
                        search_param=False, alpha=checkpoint['alpha'], beta=checkpoint['beta'])

    print("\n" + "=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"📊 Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"⚙️  Optimal Parameters: α={best_alpha:.2f}, β={best_beta:.2f}")
    print(f"🎯 Final Test Accuracy: {test_acc:.2f}%")
    print("=" * 60 + "\n")


def evaluate(model, data_loader, normalize, search_param=False, alpha=0.0, beta=1.0):
    model.eval()

    if not search_param:
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.cuda(), labels.cuda()
                images = normalize(images)
                outputs_ce, outputs_kd = model(images)

                # Convert logits to probabilities before interpolation
                probs_ce = F.softmax(outputs_ce, dim=1)
                probs_kd = F.softmax(outputs_kd / beta, dim=1)

                # Interpolate between CE and KD probabilities
                probs = alpha * probs_ce + (1 - alpha) * probs_kd

                _, predicted = torch.max(probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    # Parameter search mode
    best_acc = 0
    best_alpha = 0
    best_beta = 1.0
    alpha_range = np.arange(0, 1.05, 0.05)
    beta_range = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0])

    # Compute outputs once for all images
    all_outputs_ce = []
    all_outputs_kd = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
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
    total = len(all_labels)

    # Convert CE logits to probabilities
    all_probs_ce = F.softmax(all_outputs_ce, dim=1)

    # Parameter search
    for beta in beta_range:
        # Apply temperature scaling to KD logits
        all_probs_kd = F.softmax(all_outputs_kd / beta, dim=1)

        for alpha in alpha_range:
            # Interpolate between CE and KD probabilities
            probs = alpha * all_probs_ce + (1 - alpha) * all_probs_kd
            _, predicted = torch.max(probs.data, 1)
            correct = (predicted == all_labels).sum().item()
            acc = 100 * correct / total

            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
                best_beta = beta

    return best_acc, best_alpha, best_beta


def main():
    # Load arguments
    args = get_arguments()

    # Setup cache directory
    cache_dir = os.path.join('./logs', args.dataset)
    os.makedirs(cache_dir, exist_ok=True)

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    # Load CLIP model and teacher model
    clip_model, _ = clip.load('RN50')
    clip_model.eval()

    # Load teacher model (Tip-Adapter) if specified
    teacher_model = None
    if args.teacher_ckpt:
        print(f"Loading teacher model from {args.teacher_ckpt}")
        teacher_ckpt = torch.load(args.teacher_ckpt)

        adapter_weight = teacher_ckpt['adapter_weight']
        best_beta = teacher_ckpt['best_beta']
        best_alpha = teacher_ckpt['best_alpha']
        _ = teacher_ckpt['cache_keys']
        cache_values = teacher_ckpt['cache_values']
        clip_weights = teacher_ckpt['clip_weights']

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
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    student_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    teacher_normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    # Load dataset
    print("Preparing dataset.")
    dataset = build_dataset(args.dataset, args.root_path, args.shots)

    # Create data loaders
    train_loader_labeled = build_data_loader(
        dataset.train_x, batch_size=args.batch_size,
        tfm=train_transform, is_train=True, shuffle=True,
    )
    train_loader_unlabeled = build_data_loader(
        dataset.train_u, batch_size=args.batch_size,
        tfm=train_transform, is_train=True, shuffle=True
    )
    val_loader = build_data_loader(
        dataset.val, batch_size=args.batch_size,
        tfm=val_transform, is_train=False
    )
    test_loader = build_data_loader(
        dataset.test, batch_size=args.batch_size,
        tfm=val_transform, is_train=False
    )

    # Get CLIP weights
    clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

    # Initialize student model with model type
    student_model = StudentModel(num_classes=len(dataset.classnames), model_type=args.student_model).cuda()

    # Train student model
    train_student(args, student_model, student_normalize, clip_model, teacher_normalize, train_loader_labeled,
                  train_loader_unlabeled, val_loader, test_loader, clip_weights, teacher_model=teacher_model)


if __name__ == '__main__':
    main()
