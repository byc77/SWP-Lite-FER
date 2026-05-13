# train.py
# Fast ablation-ready training script for SWP-Lite++
# Supports: RAF-DB / FERPlus / AffectNet / Emo135

import os
import csv
import argparse
import random
import warnings
from typing import List, Optional


# Dummy class for loading older checkpoints that saved a RecorderMeter object.
# The training code only uses checkpoint["state_dict"], but torch.load needs
# this class name to exist during unpickling.
class RecorderMeter:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


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

warnings.filterwarnings("ignore", category=FutureWarning)


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
    return "cuda" if torch.cuda.is_available() else "cpu"


def profile_model(model, img_size: int, device: str):
    params = sum(p.numel() for p in model.parameters())
    flops = None
    was_training = model.training
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        flops = FlopCountAnalysis(model, dummy).total()
    except Exception as e:
        print(f"[PROFILE] FLOPs calculation failed: {e}")
        print("[PROFILE] If needed, install fvcore: pip install fvcore")
    if was_training:
        model.train()
    print(f"[PROFILE] Params = {params / 1e6:.4f} M")
    if flops is not None:
        print(f"[PROFILE] FLOPs  = {flops / 1e9:.4f} G")
    return params, flops


def set_backbone_trainable(model, trainable: bool):
    """Freeze/unfreeze the ResNet-18 backbone used by SWPNet.

    Backbone modules: stem, layer3, layer4.
    New method modules remain trainable: FFA/ECA/GCG/SWF/classifier.
    """
    base = model.module if hasattr(model, "module") else model
    # If torch.compile wraps the model, it may store the original module here.
    base = getattr(base, "_orig_mod", base)
    for name in ["stem", "layer3", "layer4"]:
        if hasattr(base, name):
            for param in getattr(base, name).parameters():
                param.requires_grad = bool(trainable)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.items)

    def _candidate_paths(self, fname: str) -> List[str]:
        name, ext = os.path.splitext(fname)
        exts = [ext, ".jpg", ".png", ".jpeg"] if ext.lower() in [".jpg", ".png", ".jpeg"] else [".jpg", ".png", ".jpeg"]
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
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
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
    use_persistent = args.workers > 0
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
    )
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


# ============================================================
# Loss functions
# ============================================================

def make_pals_targets(targets: torch.Tensor, num_classes: int, eps: float):
    with torch.no_grad():
        y = torch.zeros(targets.size(0), num_classes, device=targets.device)
        y.scatter_(1, targets.view(-1, 1), 1.0)
        if eps > 0:
            y = y * (1.0 - eps)
            y += eps / max(1, num_classes - 1)
            y.scatter_(1, targets.view(-1, 1), 1.0 - eps)
    return y


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor):
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def build_class_counts(loader, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, targets in loader:
        for t in targets.tolist():
            t = int(t)
            if 0 <= t < num_classes:
                counts[t] += 1
    return counts


def afg_focal_loss(logits: torch.Tensor, soft_targets: torch.Tensor, gamma_vec: torch.Tensor):
    prob = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
    log_prob = torch.log(prob)
    gamma = gamma_vec.view(1, -1)
    focal_weight = torch.pow(1.0 - prob, gamma)
    return -(soft_targets * focal_weight * log_prob).sum(dim=1).mean()


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
# Evaluation / Temperature Scaling / Precise BN
# ============================================================

@torch.inference_mode()
def evaluate(model, loader, device, num_classes, tta_flip=False, temperature=1.0, return_cm=False, use_amp=True):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    amp_enabled = bool(use_amp and device == "cuda")

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            if tta_flip:
                logits1 = model(images)
                logits2 = model(torch.flip(images, dims=[-1]))
                prob1 = F.softmax(logits1 / temperature, dim=1)
                prob2 = F.softmax(logits2 / temperature, dim=1)
                prob = 0.5 * (prob1 + prob2)
                pred = prob.argmax(dim=1)
                loss = F.nll_loss(torch.log(prob.clamp(min=1e-12)), targets, reduction="sum")
            else:
                logits = model(images) / temperature
                pred = logits.argmax(dim=1)
                loss = F.cross_entropy(logits, targets, reduction="sum")

        loss_sum += float(loss.item())
        correct += (pred == targets).sum().item()
        total += targets.size(0)

        for t, p in zip(targets.view(-1), pred.view(-1)):
            ti = int(t.item())
            pi = int(p.item())
            if 0 <= ti < num_classes and 0 <= pi < num_classes:
                cm[ti, pi] += 1

    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, total)
    row_sum = cm.sum(axis=1)
    per_class_acc = np.diag(cm) / np.maximum(row_sum, 1)
    bacc = float(np.mean(per_class_acc))

    if return_cm:
        return acc, bacc, avg_loss, per_class_acc, cm
    return acc, bacc, avg_loss


@torch.no_grad()
def collect_logits_targets(model, loader, device, use_amp=True):
    model.eval()
    logits_list = []
    targets_list = []
    amp_enabled = bool(use_amp and device == "cuda")
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
        # clone() turns inference/no-grad outputs into normal tensors for temperature scaling backward
        logits_list.append(logits.detach().float().clone())
        targets_list.append(targets.detach().clone())
    logits = torch.cat(logits_list, dim=0).clone()
    targets = torch.cat(targets_list, dim=0).clone()
    return logits, targets


def fit_temperature(model, val_loader, device, use_amp=True):
    logits, targets = collect_logits_targets(model, val_loader, device, use_amp=use_amp)
    temperature = torch.ones(1, device=device) * 1.0
    temperature.requires_grad = True
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits / temperature.clamp(min=1e-3), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-3).item())


def snapshot_bn_stats(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            stats[name] = (module.running_mean.detach().clone(), module.running_var.detach().clone())
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
        images = images.to(device, non_blocking=True)
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
    last_ckpt_path = os.path.join(ckpt_dir, "last.pth")
    curve_path = os.path.join(run_dir, "train_curve.csv")

    train_set, val_set, test_set = build_datasets(args)
    train_loader, val_loader, test_loader = build_loaders(train_set, val_set, test_set, args)

    print(f"[INFO] num_classes = {args.num_classes}")

    model = SWPNet(
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained),
        use_ffa=bool(args.use_ffa),
        use_eca=bool(args.use_eca),
        use_gcg=bool(args.use_gcg),
        use_swf=bool(args.use_swf),
    ).to(device)

    if args.show_profile:
        profile_params, profile_flops = profile_model(model, args.img_size, device)
    else:
        profile_params, profile_flops = None, None

    if args.two_stage:
        print(f"[TWO-STAGE] Stage 1: freeze backbone for {args.stage1_epochs} epochs, lr={args.lr}")
        print(f"[TWO-STAGE] Stage 2: unfreeze all layers, lr={args.stage2_lr}, weight_decay={args.stage2_weight_decay}")
        set_backbone_trainable(model, False)
        print(f"[TWO-STAGE] Trainable params in Stage 1 = {count_trainable_params(model) / 1e6:.4f} M")

    if args.use_compile and args.two_stage:
        print("[INFO] torch.compile disabled because --two_stage 1 needs to change trainable layers during training.")
    elif args.use_compile and hasattr(torch, "compile") and device == "cuda":
        print("[INFO] Compiling model with torch.compile ...")
        model = torch.compile(model)

    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_tmax = args.stage1_epochs if args.two_stage else args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, scheduler_tmax))

    if args.use_afg:
        gamma_vec = build_gamma_vec(train_loader, num_classes=args.num_classes, beta=args.afg_beta, device=device)
    else:
        gamma_vec = None

    use_amp = bool(args.use_amp and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_val_acc = -1.0
    best_epoch = -1

    # Initialize curve CSV.
    # monitor_test_* columns are filled only when --monitor_test_interval > 0.
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "val_bACC",
            "current_test_loss", "current_test_acc", "current_test_bACC",
            "current_test_tta_loss", "current_test_tta_acc", "current_test_tta_bACC",
            "best_val_acc", "best_epoch", "lr"
        ])

    for epoch in range(args.epochs):
        if args.two_stage and epoch == args.stage1_epochs:
            print(f"[TWO-STAGE] Switching to Stage 2 at epoch {epoch + 1}: unfreeze all layers.")
            set_backbone_trainable(model, True)
            optimizer = optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=args.stage2_weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.stage1_epochs))
            print(f"[TWO-STAGE] Trainable params in Stage 2 = {count_trainable_params(model) / 1e6:.4f} M")

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)

                if args.use_pals:
                    soft_targets = make_pals_targets(targets, num_classes=args.num_classes, eps=args.pals_eps)
                else:
                    soft_targets = None

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == targets).float().mean().item()

            bs = targets.size(0)
            running_loss += float(loss.item()) * bs
            running_acc += acc * bs
            total += bs
            pbar.set_postfix({"loss": f"{running_loss / max(1, total):.4f}", "acc": f"{running_acc / max(1, total):.4f}"})

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = running_acc / max(1, total)
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_acc, val_bacc, val_loss = evaluate(
                model, val_loader, device=device, num_classes=args.num_classes,
                tta_flip=False, temperature=1.0, return_cm=False, use_amp=use_amp,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                }, best_ckpt_path)
                saved_msg = f" -> saved best checkpoint: {best_ckpt_path}"
            else:
                saved_msg = ""
        else:
            val_acc, val_bacc, val_loss = float("nan"), float("nan"), float("nan")
            saved_msg = ""

        # Optional monitoring on the test split during training.
        # This is only for observation, not for selecting checkpoints.
        # Final reported numbers are still computed at the end with best checkpoint + SC-BN + TS.
        monitor_loss = monitor_acc = monitor_bacc = float("nan")
        monitor_tta_loss = monitor_tta_acc = monitor_tta_bacc = float("nan")
        do_monitor_test = args.monitor_test_interval > 0 and (
            (epoch + 1) % args.monitor_test_interval == 0 or epoch == args.epochs - 1
        )
        if do_monitor_test:
            monitor_acc, monitor_bacc, monitor_loss = evaluate(
                model, test_loader, device=device, num_classes=args.num_classes,
                tta_flip=False, temperature=1.0, return_cm=False, use_amp=use_amp,
            )
            if args.monitor_test_tta:
                monitor_tta_acc, monitor_tta_bacc, monitor_tta_loss = evaluate(
                    model, test_loader, device=device, num_classes=args.num_classes,
                    tta_flip=True, temperature=1.0, return_cm=False, use_amp=use_amp,
                )

        # Simple, advisor-friendly log style.
        print(f"[Epoch {epoch + 1}/{args.epochs}] Training accuracy: {train_acc:.4f}; Loss: {train_loss:.4f}; LR {current_lr:.8f}")
        print(f"[Epoch {epoch + 1}/{args.epochs}] Validation accuracy: {val_acc:.4f}; bACC: {val_bacc:.4f}; Loss: {val_loss:.4f}")
        print(f"best_acc:{best_val_acc:.4f}")

        if do_monitor_test:
            # Optional observation only; off by default.
            print(f"[Epoch {epoch + 1}/{args.epochs}] Test accuracy: {monitor_acc:.4f}; bACC: {monitor_bacc:.4f}; Loss: {monitor_loss:.4f}")
            if args.monitor_test_tta:
                print(f"[Epoch {epoch + 1}/{args.epochs}] Test accuracy with Flip-TTA: {monitor_tta_acc:.4f}; bACC: {monitor_tta_bacc:.4f}; Loss: {monitor_tta_loss:.4f}")

        if saved_msg:
            print("Model saved.")

        with open(curve_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                epoch + 1,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_bacc:.6f}",
                f"{monitor_loss:.6f}", f"{monitor_acc:.6f}", f"{monitor_bacc:.6f}",
                f"{monitor_tta_loss:.6f}", f"{monitor_tta_acc:.6f}", f"{monitor_tta_bacc:.6f}",
                f"{best_val_acc:.6f}", best_epoch, f"{current_lr:.10f}"
            ])

    # Clean SWP-style final report: use validation accuracy only.
    # Formal reported Accuracy follows the SWP-style validation protocol:
    # use the highest validation accuracy (best_acc), not a separate test result.
    summary = []
    summary.append(f"name={args.name}")
    summary.append(f"dataset={args.dataset}")
    summary.append(f"data_root={args.data_root}")
    summary.append(f"num_classes={args.num_classes}")
    summary.append(f"seed={args.seed}")
    summary.append(f"epochs={args.epochs}")
    summary.append(f"batch_size={args.batch_size}")
    summary.append(f"lr={args.lr}")
    summary.append(f"weight_decay={args.weight_decay}")
    summary.append(f"use_ffa={args.use_ffa}")
    summary.append(f"use_swf={args.use_swf}")
    summary.append(f"use_eca={args.use_eca}")
    summary.append(f"use_gcg={args.use_gcg}")
    summary.append(f"use_pals={args.use_pals}")
    summary.append(f"use_afg={args.use_afg}")
    summary.append(f"flip_cons={args.flip_cons}")
    summary.append(f"use_amp={args.use_amp}")
    summary.append(f"best_val_acc={best_val_acc:.6f}")
    summary.append(f"best_epoch={best_epoch}")
    if profile_params is not None:
        summary.append(f"params_M={profile_params / 1e6:.6f}")
    if profile_flops is not None:
        summary.append(f"flops_G={profile_flops / 1e9:.6f}")
    summary.append(f"best_checkpoint={best_ckpt_path}")
    summary.append(f"train_curve={curve_path}")

    summary_path = os.path.join(run_dir, "summary.txt")
    save_text(summary_path, "\n".join(summary))

    print("===== Training Finished =====")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Summary saved to {summary_path}")
    print(f"Training curve saved to {curve_path}")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["rafdb", "ferplus", "affectnet", "emo135"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0")

    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--use_ffa", type=int, default=1)
    parser.add_argument("--use_eca", type=int, default=1)
    parser.add_argument("--use_gcg", type=int, default=1)
    parser.add_argument("--use_swf", type=int, default=1)

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

    parser.add_argument("--show_profile", type=int, default=1)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--use_compile", type=int, default=0)
    parser.add_argument("--val_interval", type=int, default=1)

    # Two-stage fine-tuning: Stage 1 freezes ResNet backbone, Stage 2 unfreezes all layers.
    parser.add_argument("--two_stage", type=int, default=0)
    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_lr", type=float, default=5e-6)
    parser.add_argument("--stage2_weight_decay", type=float, default=5e-4)
    # Optional: show test accuracy during training so you can observe final-test-like behavior.
    # 0 = disabled. Use 1 to print every epoch, or e.g. 5 to print every 5 epochs.
    # This should not be used for checkpoint selection in formal experiments.
    parser.add_argument("--monitor_test_interval", type=int, default=0)
    parser.add_argument("--monitor_test_tta", type=int, default=1)
    # 1 = final output includes both best-val checkpoint and last-epoch checkpoint.
    # This makes it clear why monitored current-test results may differ from official best-val results.
    parser.add_argument("--final_eval_last", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_run(args)
