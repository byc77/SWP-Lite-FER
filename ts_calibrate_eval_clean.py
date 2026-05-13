# ts_calibrate_eval_clean.py
# Standalone calibration evaluation for SWP-Lite++
# Measures no-TTA Accuracy / ECE / Brier before and after Temperature Scaling.

import os
import argparse
import random
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from networks.SWP_cbpbn import SWPNet


class RecorderMeter:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu_ids: str):
    if gpu_ids == "-1":
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return "cuda" if torch.cuda.is_available() else "cpu"


class RafDataset(Dataset):
    def __init__(self, raf_path: str, phase: str = "train", img_size: int = 112):
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

        if not items:
            raise RuntimeError(f"No RAF-DB samples found. phase={phase}")

        self.items = items
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"[RAF-DB] phase={phase}, samples={len(self.items)}")
        print(f"[RAF-DB] label_file={self.label_file}")
        print(f"[RAF-DB] img_root={self.img_root}")

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
        return self.transform(img), label


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


def get_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_imagefolder_dataset(data_root: str, split: str, img_size: int):
    root = find_split_root(data_root, split)
    if root is None:
        raise FileNotFoundError(f"Cannot find split={split} folder under {data_root}")
    ds = ImageFolder(root=root, transform=get_transform(img_size))
    print(f"[ImageFolder] split={split}, root={root}, samples={len(ds)}, classes={ds.classes}")
    return ds


def build_loaders(args):
    dataset = args.dataset.lower()

    if dataset == "rafdb":
        full_train_eval = RafDataset(args.data_root, phase="train", img_size=args.img_size)
        test_set = RafDataset(args.data_root, phase="test", img_size=args.img_size)

        indices = np.arange(len(full_train_eval))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(indices)
        n_val = int(len(indices) * args.val_ratio)
        val_indices = indices[:n_val].tolist()
        train_indices = indices[n_val:].tolist()

        train_eval_set = Subset(full_train_eval, train_indices)
        val_set = Subset(full_train_eval, val_indices)
        print(f"[RAF-DB split] train_eval={len(train_eval_set)}, val={len(val_set)}, test={len(test_set)}")

    elif dataset in ["ferplus", "affectnet", "emo135"]:
        train_eval_set = build_imagefolder_dataset(args.data_root, "train", args.img_size)
        try:
            val_set = build_imagefolder_dataset(args.data_root, "valid", args.img_size)
        except Exception:
            val_set = build_imagefolder_dataset(args.data_root, "val", args.img_size)
        test_set = build_imagefolder_dataset(args.data_root, "test", args.img_size)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    use_persistent = args.workers > 0
    kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=use_persistent,
    )
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor

    train_eval_loader = DataLoader(train_eval_set, **kwargs)
    val_loader = DataLoader(val_set, **kwargs)
    test_loader = DataLoader(test_set, **kwargs)
    return train_eval_loader, val_loader, test_loader


def load_checkpoint(model, checkpoint_path, device):
    print(f"[INFO] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        nk = k.replace("_orig_mod.", "").replace("module.", "")
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[INFO] missing keys: {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")
    if missing:
        print("[WARN] first missing keys:", missing[:10])
    if unexpected:
        print("[WARN] first unexpected keys:", unexpected[:10])


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
def precise_bn_update(model, loader, num_batches, device, use_amp=True):
    if num_batches <= 0:
        return

    print(f"[SC-BN] precise BN update with {num_batches} batches")
    model.train()
    momenta = {}
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
            module.momentum = None

    amp_enabled = bool(use_amp and device == "cuda")
    n = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            model(images)
        n += 1
        if n >= num_batches:
            break

    for module, momentum in momenta.items():
        module.momentum = momentum
    model.eval()


def blend_bn_from_snapshot(model, snapshot, alpha):
    print(f"[SC-BN] alpha blend = {alpha}")
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and name in snapshot:
            old_mean, old_var = snapshot[name]
            module.running_mean.data = alpha * module.running_mean.data + (1.0 - alpha) * old_mean.to(module.running_mean.device)
            module.running_var.data = alpha * module.running_var.data + (1.0 - alpha) * old_var.to(module.running_var.device)


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
        logits_list.append(logits.detach().float().clone())
        targets_list.append(targets.detach().clone())

    return torch.cat(logits_list, dim=0).clone(), torch.cat(targets_list, dim=0).clone()


def fit_temperature_from_logits(logits, targets, device):
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


def metrics_from_logits(logits, targets, num_classes, temperature=1.0, n_bins=15):
    logits = logits.float()
    targets = targets.long()
    probs = F.softmax(logits / float(temperature), dim=1)
    conf, pred = probs.max(dim=1)

    acc = (pred == targets).float().mean().item()

    onehot = F.one_hot(targets, num_classes=num_classes).float()
    brier = torch.mean(torch.sum((probs - onehot) ** 2, dim=1)).item()

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == 0:
            in_bin = (conf >= lo) & (conf <= hi)
        else:
            in_bin = (conf > lo) & (conf <= hi)

        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = (pred[in_bin] == targets[in_bin]).float().mean()
            conf_bin = conf[in_bin].mean()
            ece += torch.abs(conf_bin - acc_bin) * prop

    return {
        "acc": acc,
        "ece": float(ece.item()),
        "brier": brier,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["rafdb", "ferplus", "affectnet", "emo135"])
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--use_ffa", type=int, default=1)
    parser.add_argument("--use_swf", type=int, default=1)
    parser.add_argument("--use_eca", type=int, default=1)
    parser.add_argument("--use_gcg", type=int, default=1)

    parser.add_argument("--precise_bn_batches", type=int, default=800)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--out_txt", type=str, default="")

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device(args.gpu_ids)
    print(f"[INFO] device = {device}")

    train_eval_loader, val_loader, test_loader = build_loaders(args)

    model = SWPNet(
        num_classes=args.num_classes,
        pretrained=bool(args.pretrained),
        use_ffa=bool(args.use_ffa),
        use_eca=bool(args.use_eca),
        use_gcg=bool(args.use_gcg),
        use_swf=bool(args.use_swf),
    ).to(device)

    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    use_amp = bool(args.use_amp and device == "cuda")

    if args.precise_bn_batches > 0:
        snap = snapshot_bn_stats(model)
        precise_bn_update(model, train_eval_loader, args.precise_bn_batches, device, use_amp=use_amp)
        blend_bn_from_snapshot(model, snap, args.alpha)

    print("[TS] collecting validation logits")
    val_logits, val_targets = collect_logits_targets(model, val_loader, device, use_amp=use_amp)
    print("[TS] fitting temperature on validation set")
    T_value = fit_temperature_from_logits(val_logits, val_targets, device)

    print("[EVAL] collecting test logits")
    test_logits, test_targets = collect_logits_targets(model, test_loader, device, use_amp=use_amp)

    before = metrics_from_logits(test_logits, test_targets, args.num_classes, temperature=1.0, n_bins=args.n_bins)
    after = metrics_from_logits(test_logits, test_targets, args.num_classes, temperature=T_value, n_bins=args.n_bins)

    lines = []
    lines.append("===== Calibration Analysis: no-TTA + SC-BN/alpha-blend + TS =====")
    lines.append(f"dataset={args.dataset}")
    lines.append(f"checkpoint={args.checkpoint}")
    lines.append(f"num_classes={args.num_classes}")
    lines.append(f"precise_bn_batches={args.precise_bn_batches}")
    lines.append(f"alpha={args.alpha}")
    lines.append(f"temperature={T_value:.6f}")
    lines.append("")
    lines.append("Setting,Acc,ECE,Brier,Temperature")
    lines.append(f"Before_TS,{before['acc']:.6f},{before['ece']:.6f},{before['brier']:.6f},-")
    lines.append(f"After_TS,{after['acc']:.6f},{after['ece']:.6f},{after['brier']:.6f},{T_value:.6f}")

    report = "\n".join(lines)
    print(report)

    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[INFO] saved to {args.out_txt}")


if __name__ == "__main__":
    main()
