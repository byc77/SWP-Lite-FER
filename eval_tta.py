# eval_tta.py
import os
import argparse
from PIL import Image
import numpy as np  # for confusion matrix / prediction statistics

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from networks.SWP_cbpbn import SWPNet
from utils.precise_bn import precise_bn_update, snapshot_bn_stats, blend_bn_from_snapshot


# ===== RafDataset：與訓練腳本共用 / 相容的 RAF-DB 載入邏輯 =====
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
        self.ten_crop = T.TenCrop(size=img_size)  # 5 crops + 5 flipped crops

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
        return img, lab


# ===== 推論輔助函式 =====
@torch.no_grad()
def logits_from_imgs(model, imgs):  # imgs: (N, 3, H, W)
    return model(imgs)


def probs_from_logits(logits):
    return F.softmax(logits, dim=1)


def eval_with_tta(model, dataset, device, batch_size=16, workers=0, tta_flip=False, tta_fivecrop=False):
    """
    回傳：
      acc: float
      y_true: np.ndarray, shape (N,)
      y_pred: np.ndarray, shape (N,)
    """
    model.eval()

    def _collate(batch):
        imgs, labels = zip(*batch)
        xs = []
        if tta_fivecrop:
            for img in imgs:
                crops = dataset.ten_crop(img)  # 10 PIL images
                crops = [dataset.normalize(dataset.to_tensor(c)) for c in crops]
                xs.append(torch.stack(crops, dim=0))  # (10,3,H,W)
        else:
            for img in imgs:
                x = dataset.normalize(dataset.to_tensor(img))  # (3,H,W)
                if tta_flip:
                    xf = torch.flip(x, dims=[2])  # horizontal flip
                    xs.append(torch.stack([x, xf], dim=0))  # (2,3,H,W)
                else:
                    xs.append(x.unsqueeze(0))  # (1,3,H,W)
        X = torch.stack(xs, dim=0)  # (B, K, 3, H, W)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate,
    )

    total, correct = 0, 0
    y_true_all, y_pred_all = [], []

    for X, y in loader:
        B, K, C, H, W = X.shape
        X = X.view(B * K, C, H, W).to(device)
        logits = logits_from_imgs(model, X)  # (B*K, C)
        probs = probs_from_logits(logits).view(B, K, -1).mean(dim=1)  # (B, C)
        pred = probs.argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += B

        # 收集每筆樣本的真值與預測，方便後續輸出 confusion matrix
        y_true_all.extend(y.numpy().tolist())
        y_pred_all.extend(pred.numpy().tolist())

    acc = correct / max(1, total)
    return acc, np.asarray(y_true_all, dtype=int), np.asarray(y_pred_all, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raf_path', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=112)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--tta_flip', type=int, default=1)
    ap.add_argument('--tta_fivecrop', type=int, default=0)        # whether to enable five-crop / ten-crop style TTA
    ap.add_argument('--precise_bn_batches', type=int, default=0)  # >0 means recalibrate BN statistics before evaluation
    ap.add_argument('--alpha', type=float, default=-1.0)          # 0~1; if >=0, blend train BN and calibrated BN
    ap.add_argument('--save_confmat', type=str, default="", help="Path to save confusion matrix CSV (optional)")  # optional output path
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    ds_test = RafDataset(args.raf_path, phase='test', img_size=args.img_size)

    # model
    model = SWPNet(num_classes=7, pretrained=False).to(device)

    # 先用一個 dummy input 跑一次 forward，避免 GCG lazy-init 後載入權重時出現 gcg.mlp.* unexpected keys
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, args.img_size, args.img_size, device=device))

    # 載入 state dict；若 checkpoint 來自 DataParallel，去掉 "module." 前綴
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print('[load] missing:', missing, 'unexpected:', unexpected)

    # Optional：先用 train set 重新估計 BN 統計量，再視需要做 alpha-blend
    if args.precise_bn_batches and args.precise_bn_batches > 0:
        ds_train = RafDataset(args.raf_path, phase='train', img_size=args.img_size)

        def _bn_collate(b):
            pil_imgs = [ds_test.resize(x[0]) for x in b]
            xs = [ds_test.normalize(ds_test.to_tensor(img)) for img in pil_imgs]
            return torch.stack(xs, dim=0), None

        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=_bn_collate,
        )
        snaps = snapshot_bn_stats(model)
        precise_bn_update(model, train_loader, num_batches=args.precise_bn_batches, device=device)
        if args.alpha >= 0.0:
            a = max(0.0, min(1.0, float(args.alpha)))
            blend_bn_from_snapshot(model, snaps, a)
            print(f'[SC-BN] blended with alpha={a:.2f}')

    # 執行 TTA 評估，回傳 acc、y_true、y_pred
    acc, y_true, y_pred = eval_with_tta(
        model,
        ds_test,
        device,
        batch_size=args.batch_size,
        workers=args.workers,
        tta_flip=bool(args.tta_flip),
        tta_fivecrop=bool(args.tta_fivecrop),
    )
    print(f'[TTA] test acc = {acc:.4f}')

    # 如果有指定輸出路徑，就把 confusion matrix 存成 CSV
    if args.save_confmat:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1) if y_true.size else 0
        C = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            C[t, p] += 1
        os.makedirs(os.path.dirname(args.save_confmat), exist_ok=True)
        np.savetxt(args.save_confmat, C, fmt="%d", delimiter=",")
        print(f'[saved] confusion matrix -> {args.save_confmat}')


if __name__ == '__main__':
    main()