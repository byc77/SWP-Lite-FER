# -*- coding: utf-8 -*-
import argparse, os, numpy as np, torch
from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

# ---- ? ----
ap = argparse.ArgumentParser()
ap.add_argument("--raf_path", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--img_size", type=int, default=112)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--alpha", type=float, default=1.0)  # ?芣靽??亙嚗???SC-BN
ap.add_argument("--out_cm", required=True)           # 瘛瑟??拚頛詨
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Dataset嚗?雿?獢??RafDataSet嚗??停韏啁陛??瑼?----
def build_loader():
    try:
        from rafdb_test import RafDataSet
        tfm = T.Compose([T.Resize((args.img_size,args.img_size)), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
        ds = RafDataSet(args.raf_path, phase='test', transform=tfm)
        print("[ds] use project RafDataSet")
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    except Exception as e:
        print(f"[ds] fallback simple loader: {e}")
        # 蝪⊥?霈瘜???list_patition_label.txt + Image 鞈?憭?
        lab = Path(args.raf_path)/"EmoLabel"/"list_patition_label.txt"
        items=[]
        with open(lab,"r",encoding="utf-8") as f:
            for line in f:
                n,l = line.strip().split()
                l = int(l)-1
                if n.lower().startswith("test"):
                    items.append((n,l))
        cand_dirs=[Path(args.raf_path)/"Image"/"aligned", Path(args.raf_path)/"Image"/"basic", Path(args.raf_path)/"Image"]
        paths,labels=[],[]
        for n,l in items:
            p=None
            for d in cand_dirs:
                q=d/n
                if q.exists():
                    p=q;break
            if p is not None:
                paths.append(p); labels.append(l)
        class _DS(torch.utils.data.Dataset):
            def __len__(self): return len(paths)
            def __getitem__(self,i):
                x=Image.open(paths[i]).convert("RGB")
                x=T.Compose([T.Resize((args.img_size,args.img_size)),T.ToTensor(),T.Normalize([0.5]*3,[0.5]*3)])(x)
                return x, labels[i]
        return DataLoader(_DS(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

# ---- Model嚗? networks.SWP_cbpbn.build_model ----
def build_model():
    try:
        from networks.SWP_cbpbn import build_model
        m = build_model(num_classes=7, img_size=args.img_size)
        print("[model] networks.SWP_cbpbn.build_model")
        return m
    except Exception as e:
        raise RuntimeError(f"隢?ㄐ?箔?撠??遣璅∠?撘?{e}")

test_loader = build_loader()
model = build_model()
ckpt = torch.load(args.checkpoint, map_location=device)
state = ckpt.get("state_dict", ckpt)
_ = model.load_state_dict(state, strict=False)
model.to(device).eval()

# ---- ?刻?銝血? CM ----
y_true=[]; y_pred=[]
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

y_true = np.asarray(y_true, dtype=int)
y_pred = np.asarray(y_pred, dtype=int)
K = int(max(y_true.max(), y_pred.max()) + 1)
C = np.zeros((K,K), dtype=int)
for t,p in zip(y_true,y_pred):
    C[t,p]+=1

Path(args.out_cm).parent.mkdir(parents=True, exist_ok=True)
np.savetxt(args.out_cm, C, fmt="%d", delimiter=",")
print("[saved] confusion:", args.out_cm)


