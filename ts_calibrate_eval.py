# -*- coding: utf-8 -*-
# ts_calibrate_eval.py — 在 RAF-DB 上進行溫度縮放 (TS) 校準
# 做法：用 train split 擬合 T（最小化 NLL），回報 test split 的 Acc、ECE、Brier（TS 前/後）
import os, argparse, json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from networks.SWP_cbpbn import SWPNet
from utils.precise_bn import precise_bn_update, snapshot_bn_stats, blend_bn_from_snapshot

# ---------------- Dataset（沿用你 eval_tta.py 的寫法） ----------------
class RafDataset(Dataset):
    def __init__(self, raf_path: str, phase: str = 'test', img_size: int = 112):
        super().__init__()
        assert phase in ['train', 'test', 'trainval']
        self.raf_path = raf_path
        self.phase = phase

        cand_labels = [
            os.path.join(raf_path, 'EmoLabel', 'list_patition_label.txt'),
            os.path.join(raf_path, 'basic', 'EmoLabel', 'list_patition_label.txt'),
            os.path.join(raf_path, 'EmoLabel', 'list_patition_label_aligned.txt'),
        ]
        label_file = next((p for p in cand_labels if os.path.isfile(p)), None)
        if label_file is None:
            raise FileNotFoundError(f'Cannot find RAF label file under: {raf_path}')

        cand_img_roots = [
            os.path.join(raf_path, 'Image', 'aligned'),
            os.path.join(raf_path, 'basic', 'Image', 'aligned'),
            os.path.join(raf_path, 'Image', 'original'),
            os.path.join(raf_path, 'basic', 'Image', 'original'),
            os.path.join(raf_path, 'Image'),
            os.path.join(raf_path),
        ]
        self.img_root = next((p for p in cand_img_roots if os.path.isdir(p)), None)
        if self.img_root is None:
            raise FileNotFoundError(f'Cannot find RAF image folder under: {raf_path}')

        items = []
        with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                fname, lab = parts[0], int(parts[1]) - 1  # 1..7 -> 0..6
                low = fname.lower()
                if phase == 'train' and 'train' in low:
                    items.append((fname, lab))
                elif phase == 'test' and 'test' in low:
                    items.append((fname, lab))
                elif phase == 'trainval':
                    items.append((fname, lab))
        if len(items) == 0:
            raise RuntimeError(f'No items for phase={phase}. Check your list file.')
        self.items = items

        self.resize = T.Resize((img_size, img_size))
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.items)

    def _candidates(self, fname: str):
        name, ext = os.path.splitext(fname)
        exts = [ext, '.jpg', '.png'] if ext.lower() in ['.jpg', '.png'] else ['.jpg', '.png']
        names = [name, f"{name}_aligned"]
        cands = []
        for nm in names:
            for ex in exts:
                b = nm + ex
                cands += [
                    os.path.join(self.img_root, b),
                    os.path.join(self.img_root, 'train', b),
                    os.path.join(self.img_root, 'test', b),
                ]
        return cands

    def _load_image(self, fname: str) -> Image.Image:
        path = next((p for p in self._candidates(fname) if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(f'Image not found for {fname} under {self.img_root}')
        img = Image.open(path).convert('RGB')
        return self.resize(img)

    def __getitem__(self, idx: int):
        fname, lab = self.items[idx]
        img = self._load_image(fname)
        x = self.normalize(self.to_tensor(img))
        return x, lab

# ---------------- Metrics: ECE / Brier ----------------
def ece_score(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    # probs: (N, C) after softmax; labels: (N,)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    for i in range(n_bins):
        start, end = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > start) & (confidences <= end)
        if mask.any():
            acc_bin = accuracies[mask].float().mean()
            conf_bin = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(conf_bin - acc_bin)
    return ece.item()

def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    # multi-class Brier
    N, C = probs.shape
    one_hot = torch.zeros_like(probs)
    one_hot[torch.arange(N, device=labels.device), labels] = 1.0
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()

# ---------------- Temperature module ----------------
class TemperatureScaler(nn.Module):
    def __init__(self, T_init: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor([T_init], dtype=torch.float32)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.logT)
        return logits / T

    def temperature(self) -> float:
        return float(torch.exp(self.logT).detach().cpu().item())

def collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            z = model(x)
            logits_all.append(z.detach().cpu())
            labels_all.append(y.detach().cpu())
    logits = torch.cat(logits_all, dim=0)   # (N, C)
    labels = torch.cat(labels_all, dim=0)   # (N,)
    return logits, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raf_path', required=True, type=str)
    ap.add_argument('--checkpoint', required=True, type=str)
    ap.add_argument('--img_size', type=int, default=112)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--precise_bn_batches', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=-1.0)
    ap.add_argument('--out_txt', type=str, default='runs/rafdb/ts_metrics.txt')
    ap.add_argument('--out_json', type=str, default='runs/rafdb/ts_metrics.json')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    ds_tr = RafDataset(args.raf_path, phase='train', img_size=args.img_size)
    ds_te = RafDataset(args.raf_path, phase='test',  img_size=args.img_size)
    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    ld_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model
    model = SWPNet(num_classes=7, pretrained=False).to(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, args.img_size, args.img_size, device=device))

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    new_state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print('[load] missing:', missing, 'unexpected:', unexpected)

    # Optional: Precise-BN + α-blend
    if args.precise_bn_batches and args.precise_bn_batches > 0:
        snaps = snapshot_bn_stats(model)
        precise_bn_update(model, ld_tr, num_batches=args.precise_bn_batches, device=device)
        if args.alpha >= 0.0:
            a = max(0.0, min(1.0, float(args.alpha)))
            blend_bn_from_snapshot(model, snaps, a)
            print(f'[SC-BN] blended with alpha={a:.2f}')

    # 1) collect logits
    logits_tr, labels_tr = collect_logits(model, ld_tr, device)
    logits_te, labels_te = collect_logits(model, ld_te, device)

    # 2) before TS metrics（on test）
    with torch.no_grad():
        probs_te = F.softmax(torch.as_tensor(logits_te), dim=1)
        acc_te = (probs_te.argmax(1) == torch.as_tensor(labels_te)).float().mean().item()
        ece_te = ece_score(probs_te, torch.as_tensor(labels_te))
        brier_te = brier_score(probs_te, torch.as_tensor(labels_te))

    # 3) fit T on train (NLL)
    scaler = TemperatureScaler(T_init=1.0).to('cpu')
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=100)
    nll = nn.CrossEntropyLoss()

    logits_tr_t = torch.as_tensor(logits_tr)
    labels_tr_t = torch.as_tensor(labels_tr, dtype=torch.long)
    def _closure():
        opt.zero_grad()
        loss = nll(scaler(logits_tr_t), labels_tr_t)
        loss.backward()
        return loss
    opt.step(_closure)
    T_star = scaler.temperature()
    print(f'[TS] fitted temperature T = {T_star:.4f}')

    # 4) after TS metrics（on test）
    with torch.no_grad():
        logits_te_cal = scaler(torch.as_tensor(logits_te))
        probs_te_cal = F.softmax(logits_te_cal, dim=1)
        acc_te_cal = (probs_te_cal.argmax(1) == torch.as_tensor(labels_te)).float().mean().item()
        ece_te_cal = ece_score(probs_te_cal, torch.as_tensor(labels_te))
        brier_te_cal = brier_score(probs_te_cal, torch.as_tensor(labels_te))

    # 5) save
    lines = [
        f"T={T_star:.4f}",
        f"beforeTS: acc={acc_te:.4f}, ECE={ece_te:.4f}, Brier={brier_te:.4f}",
        f" afterTS: acc={acc_te_cal:.4f}, ECE={ece_te_cal:.4f}, Brier={brier_te_cal:.4f}",
    ]
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({
            "T": T_star,
            "before": {"acc": acc_te, "ECE": ece_te, "Brier": brier_te},
            "after":  {"acc": acc_te_cal, "ECE": ece_te_cal, "Brier": brier_te_cal},
        }, f, ensure_ascii=False, indent=2)

    print("\n".join(lines))

if __name__ == "__main__":
    main()
