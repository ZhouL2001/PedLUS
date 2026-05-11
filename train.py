import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model import StudentModel
from classifier import VideoClassifier
from datetime import datetime
from dataset import LUSVideoClassificationDataset
from dataset_blues import build_datasets
import numpy as np
import random

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ============== FOCAL LOSS 实现 ==================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=-1)
        p = torch.exp(logp)
        ce_loss = F.nll_loss(logp, targets, weight=self.weight, reduction='none')
        focal_weight = (1 - p.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============== 原型分散正则 ==================
def prototype_separation_loss(prototypes, eps=1e-6):
    num_classes = prototypes.size(0)
    dists = torch.cdist(prototypes, prototypes)
    eye = torch.eye(num_classes, device=prototypes.device, dtype=torch.bool)
    loss = (1.0 / (dists[~eye] + eps)).mean()
    return loss


# ============== 类别权重 ==================
def compute_class_weights(dataloader, num_classes):
    class_counts = torch.zeros(num_classes)
    for _, labels, _ in dataloader:
        for label in labels:
            class_counts[label] += 1
    total_samples = class_counts.sum()
    class_weights = total_samples / class_counts
    return class_weights


# ============== 计算多分类 specificity ==================
def compute_macro_specificity(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity_list = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_list.append(spec_i)

    return float(np.mean(specificity_list)), specificity_list


# ============== 单个Epoch训练 ==================
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, sep_lambda=0.1, logit_scale=5.0):
    start_time = time.perf_counter()
    model.train()
    total_loss = 0.0
    f_l = 0.0
    s_l = 0.0

    for videos, labels, _ in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)

        margin = 0.5
        outputs[range(outputs.size(0)), labels] -= margin
        outputs = outputs * logit_scale

        ce_loss = criterion(outputs, labels)
        sep_loss = prototype_separation_loss(model.prototypes)

        loss = ce_loss
        # loss = ce_loss + sep_lambda * sep_loss

        loss.backward()
        optimizer.step()

        f_l += ce_loss.item()
        s_l += sep_loss.item()
        total_loss += loss.item()

    epoch_time = time.perf_counter() - start_time
    print(f"[ train ] Epoch {epoch}: {epoch_time:.2f}s")
    print("CrossEntropy loss: ", f_l / len(dataloader))
    print("Prototype loss: ", s_l / len(dataloader))
    return total_loss / len(dataloader)


# ============== 修改后的 evaluate 函数 ==================
def evaluate(model, dataloader, criterion, device, epoch, args, logit_scale=5.0, log_file=None):
    start_time = time.perf_counter()
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    correct_per_class = torch.zeros(args.num_classes, device=device)
    total_per_class = torch.zeros(args.num_classes, device=device)

    with torch.no_grad():
        for videos, labels, _ in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            outputs = outputs * logit_scale
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            for i in range(labels.size(0)):
                label = labels[i]
                total_per_class[label] += 1
                if preds[i] == label:
                    correct_per_class[label] += 1

    eval_time = time.perf_counter() - start_time
    print(f"[ eval ] Epoch {epoch}: {eval_time:.2f}s")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    overall_acc = accuracy_score(all_labels, all_preds)

    class_accuracy = correct_per_class / torch.clamp(total_per_class, min=1)
    per_class_acc_list = []

    for i in range(args.num_classes):
        acc_i = class_accuracy[i].item()
        correct_count = int(correct_per_class[i].item())
        total_count = int(total_per_class[i].item())
        per_class_acc_list.append(acc_i)

        log_str = f"Class {i} | Accuracy: {acc_i:.4f} ({correct_count}/{total_count})"
        print(log_str)
        if log_file is not None:
            log_file.write(log_str + '\n')
            log_file.flush()

    metric_dict = {
        "accuracy": overall_acc,
        "per_class_accuracy": per_class_acc_list
    }

    if args.dataset_name == 'blus':
        macro_precision = precision_score(
            all_labels, all_preds, average='macro', zero_division=0
        )
        macro_f1 = f1_score(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Sensitivity 一般按 macro recall 处理
        sensitivity = recall_score(
            all_labels, all_preds, average='macro', zero_division=0
        )

        specificity, specificity_per_class = compute_macro_specificity(
            all_labels, all_preds, args.num_classes
        )

        try:
            y_true_bin = label_binarize(all_labels, classes=list(range(args.num_classes)))
            macro_auc = roc_auc_score(
                y_true_bin, all_probs, average='macro', multi_class='ovr'
            )
        except Exception:
            macro_auc = float('nan')

        metric_dict.update({
            "macro_precision": macro_precision,
            "macro_f1": macro_f1,
            "macro_auc": macro_auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "specificity_per_class": specificity_per_class
        })

        metric_log = (
            f"[BLUS][Epoch {epoch}] "
            f"Accuracy: {overall_acc:.4f} | "
            f"Macro Precision: {macro_precision:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Macro AUC: {macro_auc:.4f} | "
            f"Sensitivity: {sensitivity:.4f} | "
            f"Specificity: {specificity:.4f}"
        )
        print(metric_log)
        if log_file is not None:
            log_file.write(metric_log + '\n')
            log_file.flush()

        class_acc_log = " | ".join(
            [f"Class {i} Acc: {acc:.4f}" for i, acc in enumerate(per_class_acc_list)]
        )
        print(class_acc_log)
        if log_file is not None:
            log_file.write(class_acc_log + '\n')
            log_file.flush()

    return total_loss / len(dataloader), overall_acc, metric_dict


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--enc_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip_len', type=int, default=24)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--group_lr', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--log_dir', type=str, default='classifier_logs')
    parser.add_argument('--sep_lambda', type=float, default=0.5)
    parser.add_argument('--logit_scale', type=float, default=10.0)
    parser.add_argument('--augment', type=bool, default=True)
    parser.add_argument('--SA', type=bool, default=True)
    parser.add_argument('--TA', type=bool, default=True)
    parser.add_argument('--dataset_name', type=str, default='plus')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=80)
    args = parser.parse_args()

    device = torch.device(args.device)

    exp = f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(os.path.join(args.log_dir, exp), exist_ok=True)
    log_path = os.path.join(
        args.log_dir,
        exp,
        f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    )
    log_file = open(log_path, 'w')

    # ========== Dataset ==========
    if args.dataset_name == 'plus':
        train_dataset = LUSVideoClassificationDataset(
            video_dir=args.train_dir,
            image_size=args.image_size,
            use_augmentation=args.augment
        )
        test_dataset = LUSVideoClassificationDataset(
            video_dir=args.test_dir,
            image_size=args.image_size
        )

        print(len(train_dataset))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False
        )

    elif args.dataset_name == 'blus':
        train_dataset, test_dataset = build_datasets(
            video_root='/remote-home/share/24-zhouling/workspace/dataset/COVID-BLUES/lus_videos',
            csv_path='/remote-home/share/24-zhouling/workspace/dataset/COVID-BLUES/severity.csv',
            fold=args.fold,
            num_frames=args.num_frames,
            image_size=(224, 224),
            return_metadata=False,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4
        )

        print(train_dataset.summary())
        print(test_dataset.summary())

    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    # ========== Model ==========
    student = StudentModel(embed_dim=args.embed_dim, use_temporal=True)

    if args.pretrained_path is not None:
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        student.load_state_dict(state_dict)

    encoder = student.encoder

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    model = VideoClassifier(
        encoder=encoder,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        clip_len=args.clip_len,
        TA=args.TA
    ).to(device)

    # ========== Optimizer ==========
    if args.group_lr:
        print("Group Learning Rate!")
        enc_params = {
            'params': [p for p in model.encoder.parameters() if p.requires_grad],
            'lr': args.enc_lr
        }
        head_params = {
            "params": list(model.temporal_fuser.parameters()) + list(model.classifier.parameters()),
            "lr": args.head_lr
        }
        optimizer = optim.AdamW([enc_params, head_params], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # ========== Freeze BN ==========
    def freeze_bn_stats(m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    model.encoder.apply(freeze_bn_stats)

    # ========== Loss ==========
    class_weights = compute_class_weights(train_loader, args.num_classes)
    class_weights = class_weights.to(device)
    print(f"Class Weights: {class_weights}")

    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    criterion = FocalLoss(weight=class_weights, gamma=2)

    # ========== Training Loop ==========
    best_acc = 0.0
    overall_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            sep_lambda=args.sep_lambda,
            logit_scale=args.logit_scale
        )

        test_loss, test_acc, metric_dict = evaluate(
            model,
            test_loader,
            criterion,
            device,
            epoch,
            args=args,
            logit_scale=args.logit_scale,
            log_file=log_file
        )

        if args.dataset_name == 'blus':
            log_str = (
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Acc: {metric_dict['accuracy']:.4f} | "
                f"Macro Precision: {metric_dict['macro_precision']:.4f} | "
                f"Macro F1: {metric_dict['macro_f1']:.4f} | "
                f"Macro AUC: {metric_dict['macro_auc']:.4f} | "
                f"Sensitivity: {metric_dict['sensitivity']:.4f} | "
                f"Specificity: {metric_dict['specificity']:.4f}"
            )
        else:
            log_str = (
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

        print(log_str)
        log_file.write(log_str + '\n')
        log_file.flush()

        if args.dataset_name == 'blus':
            class_log = " | ".join(
                [f"Class {i} Acc: {acc:.4f}" for i, acc in enumerate(metric_dict["per_class_accuracy"])]
            )
            print(class_log)
            log_file.write(class_log + '\n')
            log_file.flush()

        torch.save(model.state_dict(), os.path.join(args.log_dir, exp, f'{epoch}.pth'))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(args.log_dir, exp, 'best_video_classifier.pth'))
            best_str = f"[>>>] New best at Epoch {epoch} — Test Acc: {best_acc:.4f}"
            print(best_str)
            log_file.write(best_str + '\n')
            log_file.flush()

    total_time = time.perf_counter() - overall_start
    final_log = (
        f"Training complete in {total_time/60:.1f} min. "
        f"Best Test Acc: {best_acc:.4f}"
    )
    print(final_log)
    log_file.write(final_log + '\n')
    log_file.close()


if __name__ == '__main__':
    main()