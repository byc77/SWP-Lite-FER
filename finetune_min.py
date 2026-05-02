# -*- coding: utf-8 -*-
# finetune_min.py 簡化版微調腳本，預設跑 8 個 epoch，支援可選的 PALS / AFG
import os, argparse # 處理路徑 #讀命令列參數
from PIL import Image #讀圖

import torch #張量與模型
import torch.nn as nn #loss / module
import torch.nn.functional as 
from torch.utils.data import Dataset, DataLoader #資料載入
import torchvision.transforms as T #影像前處理

from networks.SWP_cbpbn import SWPNet #匯入主模型

# 嘗試載入 PALS / AFG；如果不存在，後面會 fallback 到 CE
PALS = None
FocalAFG = None
try:
    # 嘗試載入外部實作的 PALS
    from losses.pals import PALS as PALS_IMPL
    PALS = PALS_IMPL
except Exception:
    pass
try:
    # 嘗試載入外部實作的 AFG
    from losses.focal_afg import FocalAFG as FocalAFG_IMPL
    FocalAFG = FocalAFG_IMPL
except Exception:
    pass 


# ====== 與 eval 腳本共用／相容的 RAF-DB 載入邏輯 ======
class RafDataset(Dataset):
    def __init__(self, raf_path: str, phase: str = 'train', img_size: int = 112):
        super().__init__()
        assert phase in ['train', 'test', 'trainval']
        self.raf_path = raf_path
        self.phase = phase
        
        #找 label file
        cand_labels = [
            os.path.join(raf_path, 'EmoLabel', 'list_patition_label.txt'),
            os.path.join(raf_path, 'basic', 'EmoLabel', 'list_patition_label.txt'),
            os.path.join(raf_path, 'EmoLabel', 'list_patition_label_aligned.txt'),
        ]
        label_file = next((p for p in cand_labels if os.path.isfile(p)), None)
        if label_file is None:
            raise FileNotFoundError(f'Cannot find RAF label file under: {raf_path}')
        #找 image root
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
        #讀 label file，切 train / test
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
            raise RuntimeError(f'No items for phase={phase}.')
        self.items = items
        #資料增強與前處理
        aug = []
        if phase == 'train':
            aug += [T.RandomHorizontalFlip(p=0.5)]
        aug += [T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.5]*3, [0.5]*3)]
        self.tfm = T.Compose(aug)

    def __len__(self): return len(self.items) #回傳資料集總長度
    # 根據一個檔名列出所有可能的圖片實際位置。
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
    #訓練和測試時 DataLoader 真正拿到的資料
    def __getitem__(self, idx: int):
        fname, y = self.items[idx]
        path = next((p for p in self._candidates(fname) if os.path.isfile(p)), None)
        if path is None:
            raise FileNotFoundError(f'Image not found for {fname}')
        x = Image.open(path).convert('RGB')
        return self.tfm(x), y


@torch.no_grad() 
#在 validation / test loader 上算 accuracy
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def build_loss(use_pals_afg_flag: int, use_pals_flag: int, use_afg_flag: int):
    """
    回傳 (criterion, desc)
    1) 如果 use_pals_afg=1，則同時啟用 PALS 與 AFG
    2) 否則依照 use_pals / use_afg 的設定個別決定
    3) 如果外部實作不存在，則使用 fallback：
       - PALS 不存在時，改用 CE(label_smoothing=0.05)
       - AFG 不存在時，改用 CE 或僅保留可用的 loss
       - 都沒有時，使用標準 CE
    """
    # 相容模式：一鍵啟用 PALS + AFG
    if use_pals_afg_flag == 1:
        use_pals_flag, use_afg_flag = 1, 1

    have_pals = PALS is not None
    have_afg  = FocalAFG is not None

    # Case A：同時使用 PALS 與 AFG
    if use_pals_flag and use_afg_flag:
        pals_obj = None
        if have_pals:
            # 嘗試建立 PALS，並傳入 num_classes / eps 參數
            pals_obj = PALS(num_classes=7, eps=0.1)
        afg_obj = None
        if have_afg:
            afg_obj = FocalAFG(num_classes=7, gamma_base=2.0)

        if pals_obj and afg_obj:
            def loss_fn(logits, y):
                return pals_obj(logits, y) + afg_obj(logits, y)
            return loss_fn, "PALS + AFG"
        elif pals_obj and (not afg_obj):
            # AFG 缺失時，退回只使用 PALS
            def loss_fn(logits, y): return pals_obj(logits, y)
            return loss_fn, "PALS (AFG missing)"
        elif (not pals_obj) and afg_obj:
            def loss_fn(logits, y): return afg_obj(logits, y)
            return loss_fn, "AFG (PALS missing)"
        else:
            # 如果兩者都缺失，則用 label smoothing 版本的 CE 當作 PALS 替代
            return nn.CrossEntropyLoss(label_smoothing=0.05), "CE (fallback, no PALS/AFG)"

    # Case B：只使用 PALS
    if use_pals_flag and not use_afg_flag:
        if have_pals:
            pals_obj = PALS(num_classes=7, eps=0.1)
            def loss_fn(logits, y): return pals_obj(logits, y)
            return loss_fn, "PALS"
        else:
            return nn.CrossEntropyLoss(label_smoothing=0.05), "CE (PALS-fallback)"

    #Case C：只使用 AFG
    if use_afg_flag and not use_pals_flag:
        if have_afg:
            afg_obj = FocalAFG(num_classes=7, gamma_base=2.0)
            def loss_fn(logits, y): return afg_obj(logits, y)
            return loss_fn, "AFG"
        else:
            return nn.CrossEntropyLoss(), "CE (AFG-missing)"

    # Case D：都不啟用，使用 baseline CE
    return nn.CrossEntropyLoss(), "CrossEntropy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raf_path', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--img_size', type=int, default=112)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    # 可選的 loss 啟用參數
    ap.add_argument('--use_pals', type=int, default=0)
    ap.add_argument('--use_afg',  type=int, default=0)
    # 透過參數決定是否啟用 PALS / AFG
    ap.add_argument('--use_pals_afg', type=int, default=0, help="(compat) 1 = PALS+AFG")
    ap.add_argument('--out_ckpt', type=str, default='checkpoints/finetune_min.pth')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)

    ds_tr = RafDataset(args.raf_path, phase='train', img_size=args.img_size)
    ds_te = RafDataset(args.raf_path, phase='test',  img_size=args.img_size)
    ld_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    ld_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)

    # Model & load
    model = SWPNet(num_classes=7, pretrained=False).to(device)
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, args.img_size, args.img_size, device=device))
    state = torch.load(args.checkpoint, map_location=device)
    state = state['model'] if isinstance(state, dict) and 'model' in state else state
    new_state = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print('[load] missing:', missing, 'unexpected:', unexpected)

    # 只訓練分類頭，採用 head-only fine-tuning
    for n, p in model.named_parameters():
        p.requires_grad = ('fc' in n) or ('classifier' in n) or ('head' in n)
    params = [p for p in model.parameters() if p.requires_grad]

    # 建立 loss，並處理 fallback 邏輯
    loss_fn, loss_desc = build_loss(args.use_pals_afg, args.use_pals, args.use_afg)
    print(f"[loss] {loss_desc}")

    opt = torch.optim.AdamW(params, lr=args.lr)
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in ld_tr:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        acc_te = evaluate(model, ld_te, device)
        print(f"[Epoch {ep}] test acc={acc_te:.4f}")
        if acc_te > best:
            best = acc_te
            torch.save({'model': model.state_dict()}, args.out_ckpt)
            print("[save]", args.out_ckpt)

if __name__ == "__main__":
    main()


