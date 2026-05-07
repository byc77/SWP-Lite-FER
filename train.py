# train.py
# Unified training script for SWP-Lite++
# Supports: RAF-DB / FERPlus / AffectNet / Emo135
#
# Example:
# python train.py ^
# --name rafdb_swp_litepp_seed123 ^
# --dataset rafdb ^
# --data_root "D:\SWP-Stages-Weighted-Pooling-CNN-with-FER-main\Datasets\RAF-DB" ^
# --num_classes 7 ^
# --epochs 8 ^
# --batch_size 64 ^
# --img_size 112 ^
# --lr 8e-5 ^
# --weight_decay 1e-4 ^
# --seed 123 ^
# --gpu_ids 0 ^
# --pretrained 1 ^
# --use_eca 1 ^
# --use_gcg 1 ^
# --use_pals 1 ^
# --pals_eps 0.1 ^
# --use_afg 1 ^
# --afg_beta 1.0 ^
# --flip_cons 1 ^
# --lambda_cons 1.0 ^
# --scbn_batches 800 ^
# --alpha 0.75 ^
# --tta_flip 1 ^
# --use_ts 1

import os
import argparse
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from tqdm import tqdm

from networks.SWP_cbpbn import SWPNet


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def get_device(gpu_ids: str):
    if gpu_ids == "-1":
        return "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# RAF-DB Dataset
# ============================================================

class RafDataset(Dataset):
    def __init__(self, raf_path: str, phase: str = "train", img_size: int = 112, train_aug: bool = True):
        super().__init__()

        assert phase in ["train", "test"]

        self.raf_path = raf_path
        self.phase = phase

        label_candidates = [
            os.path.join(raf_path, "EmoLabel", "list_patition_label.txt"),
            os.path.join(raf_path, "basic", "EmoLabel", "list_patition_label.txt"),
            os.path.join(raf_path, "EmoLabel", "list_patition_label_aligned.txt"),
        ]

        self.label_file = next((p for p in label_candidates if os.path.isfile(p)), None)

        if self.label_file is None:
            raise FileNotFoundError(f"Cannot find RAF-DB label file under: {raf_path}")

        image_root_candidates = [
            os.path.join(raf_path, "Image", "aligned"),
            os.path.join(raf_path, "basic", "Image", "aligned"),
            os.path.join(raf_path, "Image", "original"),
            os.path.join(raf_path, "basic", "Image", "original"),
            os.path.join(raf_path, "Image"),
            raf_path,
        ]

        self.img_root = next((p for p in image_root_candidates if os.path.isdir(p)), None)

        if self.img_root is None:
            raise FileNotFoundError(f"Cannot find RAF-DB image folder under: {raf_path}")

        items = []

        with open(self.label_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 2:
                    continue

                fname = parts[0]
                label = int(parts[1]) - 1

                low = fname.lower()

                if phase == "train" and "train" in low:
                    items.append((fname, label))
                elif phase == "test" and "test" in low:
                    items.append((fname, label))

        if len(items) == 0:
            raise RuntimeError(f"No RAF-DB samples found. phase={phase}")

        self.items = items

        print(f"[RAF-DB] phase={phase}, samples={len(self.items)}")
        print(f"[RAF-DB] label_file={self.label_file}")
        print(f"[RAF-DB] img_root={self.img_root}")

        if phase == "train" and train_aug:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.items)

    def _candidate_paths(self, fname: str) -> List[str]:
        name, ext = os.path.splitext(fname)

        if ext.lower() in [".jpg", ".png", ".jpeg"]:
            exts = [ext, ".jpg", ".png", ".jpeg"]
        else:
            exts = [".jpg", ".png", ".jpeg"]

        names = [name, f"{name}_aligned"]

        paths = []

        for nm in names:
            for ex in exts:
                b = nm + ex
                paths.append(os.path.join(self.img_root, b))
                paths.append(os.path.join(self.img_root, "train", b))
                paths.append(os.path.join(self.img_root, "test", b))

        return paths

    def _load_image(self, fname: str) -> Image.Image:
        path = next((p for p in self._candidate_paths(fname) if os.path.isfile(p)), None)

        if path is None:
            raise FileNotFoundError(f"Image not found for {fname} under {self.img_root}")

        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img = self._load_image(fname)
        img = self.transform(img)
        return img, label


# ============================================================
# Generic ImageFolder Dataset
# FERPlus / AffectNet / Emo135
# ============================================================

def find_split_root(data_root: str, split: str) -> Optional[str]:
    split = split.lower()

    if split == "train":
        candidates = ["train", "Train", "training", "Training"]
    elif split in ["val", "valid", "validation"]:
        candidates = ["valid", "Valid", "val", "Val", "validation", "Validation"]
    elif split == "test":
        candidates = ["test", "Test", "testing", "Testing"]
    else:
        candidates = [split]

    for name in candidates:
        p = os.path.join(data_root, name)

        if os.path.isdir(p):
            return p

    return None


def get_transform(img_size: int, train_aug: bool):
    if train_aug:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
        ])

    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])


def build_imagefolder_dataset(data_root: str, split: str, img_size: int, train_aug: bool):
    root = find_split_root(data_root, split)

    if root is None:
        raise FileNotFoundError(f"Cannot find split={split} folder under {data_root}")

    ds = ImageFolder(root=root, transform=get_transform(img_size, train_aug=train_aug))

    print(f"[ImageFolder] split={split}, root={root}")
    print(f"[ImageFolder] samples={len(ds)}")
    print(f"[ImageFolder] classes={ds.classes}")

    return ds


# ============================================================
# Build datasets
# ============================================================

def build_datasets(args):
    dataset = args.dataset.lower()

    if dataset == "rafdb":
        full_train_aug = RafDataset(args.data_root, phase="train", img_size=args.img_size, train_aug=True)
        full_train_eval = RafDataset(args.data_root, phase="train", img_size=args.img_size, train_aug=False)
        test_set = RafDataset(args.data_root, phase="test", img_size=args.img_size, train_aug=False)

        indices = np.arange(len(full_train_aug))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)

        n_val = int(len(indices) * args.val_ratio)
        val_indices = indices[:n_val].tolist()
        train_indices = indices[n_val:].tolist()

        train_set = Subset(full_train_aug, train_indices)
        val_set = Subset(full_train_eval, val_indices)

        print(f"[RAF-DB split] train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

        return train_set, val_set, test_set

    if dataset in ["ferplus", "affectnet", "emo135"]:
        train_set = build_imagefolder_dataset(args.data_root, "train", args.img_size, train_aug=True)

        try:
            val_set = build_imagefolder_dataset(args.data_root, "valid", args.img_size, train_aug=False)
        except Exception:
            try:
                val_set = build_imagefolder_dataset(args.data_root, "val", args.img_size, train_aug=False)
            except Exception:
                print("[Warning] validation folder not found. Use test set as validation.")
                val_set = build_imagefolder_dataset(args.data_root, "test", args.img_size, train_aug=False)

        test_set = build_imagefolder_dataset(args.data_root, "test", args.img_size, train_aug=False)

        return train_set, val_set, test_set

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_loaders(train_set, val_set, test_set, args):
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Loss functions: PALS / AFG
# ============================================================

def make_pals_targets(targets: torch.Tensor, num_classes: int, eps: float):
    """
    Simple PALS-compatible sparse soft target.
    For reproducible code, this version distributes eps to all non-ground-truth classes.
    """
    with torch.no_grad():
        y = torch.zeros(targets.size(0), num_classes, device=targets.device)
        y.scatter_(1, targets.view(-1, 1), 1.0)

        if eps > 0:
            y = y * (1.0 - eps)
            y += eps / max(1, num_classes - 1)

            # correct class should not receive extra eps
            y.scatter_(1, targets.view(-1, 1), 1.0 - eps)

    return y


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor):
    log_prob = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_prob).sum(dim=1)
    return loss.mean()


def build_class_counts(loader, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.int64)

    for _, targets in loader:
        for t in targets.tolist():
            t = int(t)

            if 0 <= t < num_classes:
                counts[t] += 1

    return counts


def afg_focal_loss(logits: torch.Tensor, soft_targets: torch.Tensor, gamma_vec: torch.Tensor):
    """
    Class-adaptive focal loss for soft targets.
    gamma_vec shape: (num_classes,)
    """
    prob = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
    log_prob = torch.log(prob)

    gamma = gamma_vec.view(1, -1)
    focal_weight = torch.pow(1.0 - prob, gamma)

    loss = -(soft_targets * focal_weight * log_prob).sum(dim=1)
    return loss.mean()


def build_gamma_vec(train_loader, num_classes: int, beta: float, device: str):
    counts = build_class_counts(train_loader, num_classes)
    freq = counts / max(1, counts.sum())
    max_freq = freq.max() if freq.max() > 0 else 1.0

    gamma = 2.0 + beta * (1.0 - freq / max_freq)
    gamma_tensor = torch.tensor(gamma, dtype=torch.float32, device=device)

    print(f"[AFG] class counts = {counts.tolist()}")
    print(f"[AFG] gamma = {gamma.tolist()}")

    return gamma_tensor


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, num_classes, tta_flip=False, temperature=1.0, return_cm=False):
    model.eval()

    total = 0
    correct = 0

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if tta_flip:
            logits1 = model(images)
            logits2 = model(torch.flip(images, dims=[-1]))

            prob1 = F.softmax(logits1 / temperature, dim=1)
            prob2 = F.softmax(logits2 / temperature, dim=1)

            prob = 0.5 * (prob1 + prob2)
            pred = prob.argmax(dim=1)
        else:
            logits = model(images) / temperature
            pred = logits.argmax(dim=1)

        correct += (pred == targets).sum().item()
        total += targets.size(0)

        for t, p in zip(targets.view(-1), pred.view(-1)):
            ti = int(t.item())
            pi = int(p.item())

            if 0 <= ti < num_classes and 0 <= pi < num_classes:
                cm[ti, pi] += 1

    acc = correct / max(1, total)

    row_sum = cm.sum(axis=1)
    per_class_acc = np.diag(cm) / np.maximum(row_sum, 1)
    bacc = float(np.mean(per_class_acc))

    if return_cm:
        return acc, bacc, per_class_acc, cm

    return acc, bacc


# ============================================================
# Temperature Scaling
# ============================================================

def collect_logits_targets(model, loader, device):
    model.eval()

    logits_list = []
    targets_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            logits_list.append(logits.detach())
            targets_list.append(targets.detach())

    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)

    return logits, targets


def fit_temperature(model, val_loader, device):
    logits, targets = collect_logits_targets(model, val_loader, device)

    temperature = torch.ones(1, device=device) * 1.0
    temperature.requires_grad = True

    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature.clamp(min=1e-3), targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    T_value = float(temperature.detach().clamp(min=1e-3).item())

    return T_value


# ============================================================
# Precise-BN
# ============================================================

def snapshot_bn_stats(model):
    stats = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            stats[name] = (
                module.running_mean.detach().clone(),
                module.running_var.detach().clone(),
            )

    return stats


@torch.no_grad()
def precise_bn_update(model, loader, num_batches, device):
    if num_batches <= 0:
        return

    model.train()

    momenta = {}

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
            module.momentum = None

    n = 0

    for images, _ in loader:
        images = images.to(device)
        model(images)

        n += 1

        if n >= num_batches:
            break

    for module, momentum in momenta.items():
        module.momentum = momentum

    model.eval()


def blend_bn_from_snapshot(model, snapshot, alpha):
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and name in snapshot:
            old_mean, old_var = snapshot[name]

            module.running_mean.data = alpha * module.running_mean.data + (1.0 - alpha) * old_mean.to(module.running_mean.device)
            module.running_var.data = alpha * module.running_var.data + (1.0 - alpha) * old_var.to(module.running_var.device)


# ============================================================
# Training
# ============================================================

def train_one_run(args):
    set_seed(args.seed)

    device = get_device(args.gpu_ids)

    print(f"[INFO] device = {device}")
    print(f"[INFO] experiment name = {args.name}")
    print(f"[INFO] dataset = {args.dataset}")
    print(f"[INFO] data_root = {args.data_root}")

    if args.num_classes <= 0:
        if args.dataset.lower() == "rafdb":
            args.num_classes = 7
        elif args.dataset.lower() == "ferplus":
            args.num_classes = 8
        else:
            raise ValueError("Please specify --num_classes for AffectNet / Emo135.")

    ckpt_dir = os.path.join("checkpoints", args.name)
    run_dir = os.path.join("runs", args.name)

    make_dir(ckpt_dir)
    make_dir(run_dir)

    best_ckpt_path = os.path.join(ckpt_dir, "best.pth")

    train_set, val_set, test_set = build_datasets(args)
    train_loader, val_loader, test_loader = build_loaders(train_set, val_set, test_set, args)

    print(f"[INFO] num_classes = {args.num_classes}")

    model = SWPNet(
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained),
        use_eca=bool(args.use_eca),
        use_gcg=bool(args.use_gcg),
    ).to(device)

    # Trigger lazy modules such as GCG.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        _ = model(dummy)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    gamma_vec = build_gamma_vec(
        train_loader,
        num_classes=args.num_classes,
        beta=args.afg_beta,
        device=device,
    )

    best_val_acc = -1.0

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(images)

            if args.use_pals:
                soft_targets = make_pals_targets(
                    targets,
                    num_classes=args.num_classes,
                    eps=args.pals_eps,
                )
            else:
                soft_targets = torch.zeros(
                    targets.size(0),
                    args.num_classes,
                    device=device,
                )
                soft_targets.scatter_(1, targets.view(-1, 1), 1.0)

            if args.use_afg:
                loss_cls = afg_focal_loss(logits, soft_targets, gamma_vec)
            else:
                if args.use_pals:
                    loss_cls = soft_cross_entropy(logits, soft_targets)
                else:
                    loss_cls = F.cross_entropy(logits, targets)

            if args.flip_cons:
                images_flip = torch.flip(images, dims=[-1])
                logits_flip = model(images_flip)
                loss_cons = torch.mean((logits - logits_flip) ** 2)
                loss = loss_cls + args.lambda_cons * loss_cons
            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == targets).float().mean().item()

            bs = targets.size(0)

            running_loss += loss.item() * bs
            running_acc += acc * bs
            total += bs

            pbar.set_postfix({
                "loss": f"{running_loss / max(1, total):.4f}",
                "acc": f"{running_acc / max(1, total):.4f}",
            })

        scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = running_acc / max(1, total)

        val_acc, val_bacc = evaluate(
            model,
            val_loader,
            device=device,
            num_classes=args.num_classes,
            tta_flip=False,
            temperature=1.0,
            return_cm=False,
        )

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_bACC={val_bacc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "args": vars(args),
            }, best_ckpt_path)

            print(f"  -> saved best checkpoint: {best_ckpt_path}")

    print("[INFO] load best checkpoint for final evaluation")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    if args.scbn_batches > 0:
        print("[SC-BN] precise BN update")

        snap = snapshot_bn_stats(model)

        precise_bn_update(
            model,
            train_loader,
            num_batches=args.scbn_batches,
            device=device,
        )

        blend_bn_from_snapshot(model, snap, alpha=args.alpha)

        print(f"[SC-BN] alpha blend = {args.alpha}")

    temperature = 1.0

    if args.use_ts:
        print("[TS] fitting temperature on validation set")
        temperature = fit_temperature(model, val_loader, device)
        print(f"[TS] learned T = {temperature:.4f}")

    acc_no, bacc_no, per_no, cm_no = evaluate(
        model,
        test_loader,
        device=device,
        num_classes=args.num_classes,
        tta_flip=False,
        temperature=temperature,
        return_cm=True,
    )

    acc_tta, bacc_tta, per_tta, cm_tta = evaluate(
        model,
        test_loader,
        device=device,
        num_classes=args.num_classes,
        tta_flip=bool(args.tta_flip),
        temperature=temperature,
        return_cm=True,
    )

    print(f"[TEST] no-TTA  acc={acc_no:.4f} bACC={bacc_no:.4f}")
    print(f"[TEST] flipTTA acc={acc_tta:.4f} bACC={bacc_tta:.4f}")

    cm_no_path = os.path.join(run_dir, "cm_noTTA.csv")
    cm_tta_path = os.path.join(run_dir, "cm_TTA.csv")

    np.savetxt(cm_no_path, cm_no, fmt="%d", delimiter=",")
    np.savetxt(cm_tta_path, cm_tta, fmt="%d", delimiter=",")

    print(f"[save] confusion matrix no-TTA  -> {cm_no_path}")
    print(f"[save] confusion matrix flipTTA -> {cm_tta_path}")

    summary = []
    summary.append(f"name={args.name}")
    summary.append(f"dataset={args.dataset}")
    summary.append(f"data_root={args.data_root}")
    summary.append(f"num_classes={args.num_classes}")
    summary.append(f"seed={args.seed}")
    summary.append(f"best_val_acc={best_val_acc:.6f}")
    summary.append(f"temperature={temperature:.6f}")
    summary.append(f"noTTA_acc={acc_no:.6f}")
    summary.append(f"noTTA_bACC={bacc_no:.6f}")
    summary.append(f"TTA_acc={acc_tta:.6f}")
    summary.append(f"TTA_bACC={bacc_tta:.6f}")
    summary.append(f"best_checkpoint={best_ckpt_path}")

    summary_text = "\n".join(summary)

    summary_path = os.path.join(run_dir, "summary.txt")
    save_text(summary_path, summary_text)

    print(f"[save] summary -> {summary_path}")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["rafdb", "ferplus", "affectnet", "emo135"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu_ids", type=str, default="0")

    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--use_eca", type=int, default=1)
    parser.add_argument("--use_gcg", type=int, default=1)

    parser.add_argument("--use_pals", type=int, default=1)
    parser.add_argument("--pals_eps", type=float, default=0.1)

    parser.add_argument("--use_afg", type=int, default=1)
    parser.add_argument("--afg_beta", type=float, default=1.0)

    parser.add_argument("--flip_cons", type=int, default=1)
    parser.add_argument("--lambda_cons", type=float, default=1.0)

    parser.add_argument("--scbn_batches", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--tta_flip", type=int, default=1)
    parser.add_argument("--use_ts", type=int, default=1)

    parser.add_argument("--val_ratio", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_run(args)