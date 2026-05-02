# cmp_backbone_head8.py
import os, argparse, math, csv
from pathlib import Path
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tv

from utils.precise_bn import precise_bn_update, snapshot_bn_stats, blend_bn_from_snapshot

# -------------------- RAF Dataset：與 eval_tta 共用 / 相容的資料載入邏輯 --------------------
class RafDataset(Dataset):
    def __init__(self, raf_path, phase='train', img_size=112):
        assert phase in ['train','test','trainval']
        self.raf_path = raf_path
        self.phase = phase
        cand_labels = [
            os.path.join(raf_path,'EmoLabel','list_patition_label.txt'),
            os.path.join(raf_path,'basic','EmoLabel','list_patition_label.txt'),
            os.path.join(raf_path,'EmoLabel','list_patition_label_aligned.txt'),
        ]
        label_file = next((p for p in cand_labels if os.path.isfile(p)), None)
        if label_file is None:
            raise FileNotFoundError(f'Label file not found under: {raf_path}')
        cand_img_roots = [
            os.path.join(raf_path,'Image','aligned'),
            os.path.join(raf_path,'basic','Image','aligned'),
            os.path.join(raf_path,'Image','original'),
            os.path.join(raf_path,'basic','Image','original'),
            os.path.join(raf_path,'Image'),
            raf_path,
        ]
        self.img_root = next((p for p in cand_img_roots if os.path.isdir(p)), None)
        if self.img_root is None:
            raise FileNotFoundError(f'Image folder not found under: {raf_path}')
        items=[]
        with open(label_file,'r',encoding='utf-8',errors='ignore') as f:
            for line in f:
                ss=line.strip().split()
                if len(ss)<2: continue
                fname, lab = ss[0], int(ss[1])-1
                low=fname.lower()
                if phase=='train' and 'train'in low: items.append((fname,lab))
                elif phase=='test' and 'test'in low: items.append((fname,lab))
                elif phase=='trainval': items.append((fname,lab))
        if len(items)==0: raise RuntimeError(f'No items for phase={phase}')
        self.items=items
        self.resize=T.Resize((img_size,img_size))
        self.to_tensor=T.ToTensor()
        self.normalize=T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        self.ten_crop=T.TenCrop(img_size)

    def __len__(self): return len(self.items)

    def _cands(self,fname):
        name,ext=os.path.splitext(fname)
        exts=[ext,'.jpg','.png'] if ext.lower() in ['.jpg','.png'] else ['.jpg','.png']
        names=[name, f"{name}_aligned"]
        c=[]
        for nm in names:
            for ex in exts:
                b=nm+ex
                c+=[
                    os.path.join(self.img_root,b),
                    os.path.join(self.img_root,'train',b),
                    os.path.join(self.img_root,'test',b),
                ]
        return c

    def _open(self,fname):
        p=next((p for p in self._cands(fname) if os.path.isfile(p)), None)
        if p is None: raise FileNotFoundError(f'Image not found for {fname}')
        return Image.open(p).convert('RGB')

    def __getitem__(self,idx):
        fname,lab=self.items[idx]
        img=self.resize(self._open(fname))
        x=self.normalize(self.to_tensor(img))
        return x, lab

# -------------------- 建立比較用 backbone：mobilenet_v3_small / efficientnet_b0 / resnet18 --------------------
def build_model(arch='mobilenet_v3_small', num_classes=7, pretrained=True):
    if arch=='mobilenet_v3_small':
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        in_ch = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_ch, num_classes)
        return m
    elif arch=='efficientnet_b0':
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_ch = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_ch, num_classes)
        return m
    elif arch=='resnet18':
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        in_ch = m.fc.in_features
        m.fc = nn.Linear(in_ch, num_classes)
        return m
    else:
        raise ValueError(f'Unknown arch: {arch}')

# -------------------- 凍結 backbone，只保留 head-only 微調 --------------------
def freeze_backbone_head_only(m, arch):
    for p in m.parameters(): p.requires_grad=False
    # 只開最後分類層參數
    if arch in ['mobilenet_v3_small','efficientnet_b0']:
        for p in m.classifier[-1].parameters(): p.requires_grad=True
        params = [p for p in m.classifier[-1].parameters() if p.requires_grad]
    elif arch=='resnet18':
        for p in m.fc.parameters(): p.requires_grad=True
        params = [p for p in m.fc.parameters() if p.requires_grad]
    return params

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    total=0; correct=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        logits=model(x)
        pred=logits.argmax(1)
        correct+= (pred==y).sum().item()
        total+= y.numel()
    return correct/max(1,total)

@torch.no_grad()
def eval_tta_acc(model, ds, device, batch_size=64, workers=0, flip=True):
    # 簡單 TTA：原圖 + 水平翻轉
    def _collate(batch):
        xs, ys = [], []
        for x,y in batch:
            if flip:
                xf=torch.flip(x, dims=[2])
                xs.append(torch.stack([x,xf],0))
            else:
                xs.append(x.unsqueeze(0))
            ys.append(y)
        X=torch.stack(xs,0)   # (B, K, 3, H, W)
        Y=torch.tensor(ys, dtype=torch.long)
        return X,Y
    loader=DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=_collate)
    model.eval()
    tot=0; corr=0
    for X,Y in loader:
        B,K,C,H,W=X.shape
        X=X.view(B*K,C,H,W).to(device)
        Y=Y.to(device)
        logits=model(X).view(B,K,-1).mean(1)
        pred=logits.argmax(1)
        corr+=(pred==Y).sum().item()
        tot+=B
    return corr/max(1,tot)

@torch.no_grad()
def dump_confusion(model, loader, num_classes=7, device='cuda'):
    model.eval()
    cm=torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        pred=model(x).argmax(1)
        for t,p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()]+=1
    return cm.cpu().numpy()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raf_path', type=str, required=True)
    ap.add_argument('--arch', type=str, default='mobilenet_v3_small',
                    choices=['mobilenet_v3_small','efficientnet_b0','resnet18'])
    ap.add_argument('--img_size', type=int, default=112)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--pretrained', type=int, default=1)  # 1 = 使用 ImageNet 預訓練
    ap.add_argument('--precise_bn_batches', type=int, default=800)
    ap.add_argument('--alpha', type=float, default=0.75)
    ap.add_argument('--out_dir', type=str, default='runs/rafdb/cmp')
    ap.add_argument('--save_ckpt', type=str, default=None)
    ap.add_argument('--eval_tta', type=int, default=1)
    args=ap.parse_args()

    dev='cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # data
    ds_tr=RafDataset(args.raf_path, 'train', args.img_size)
    ds_te=RafDataset(args.raf_path, 'test', args.img_size)
    tr_loader=DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    te_loader=DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # model
    model=build_model(args.arch, num_classes=7, pretrained=bool(args.pretrained)).to(dev)
    params=freeze_backbone_head_only(model, args.arch)
    opt=torch.optim.Adam(params, lr=args.lr)
    crit=nn.CrossEntropyLoss()

    # train head-only 8 epochs
    for ep in range(1, args.epochs+1):
        model.train()
        for x,y in tr_loader:
            x=x.to(dev); y=y.to(dev)
            logits=model(x)
            loss=crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
        acc=eval_acc(model, te_loader, dev)
        print(f'[Epoch {ep}] test acc={acc:.4f}')

    # optional: save ckpt
    if args.save_ckpt:
        torch.save(model.state_dict(), args.save_ckpt)
        print(f'[save] {args.save_ckpt}')

    # Precise-BN + alpha-blend：先用 train set 重新估計 BN 統計量
    snaps=snapshot_bn_stats(model)
    def _bn_collate(b):
        xs=[x for x,_ in b]
        return torch.stack(xs,0), None
    bn_loader=DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=_bn_collate)
    precise_bn_update(model, bn_loader, num_batches=args.precise_bn_batches, device=dev)
    blend_bn_from_snapshot(model, snaps, max(0.0, min(1.0, float(args.alpha))))
    print(f'[SC-BN] blended with alpha={args.alpha:.2f}')

    # 評估：no-TTA + TTA
    acc_notta=eval_acc(model, te_loader, dev)
    print(f'[noTTA] acc={acc_notta:.4f}')
    if args.eval_tta:
        acc_tta=eval_tta_acc(model, ds_te, dev, batch_size=args.batch_size, workers=args.workers, flip=True)
        print(f'[TTA] acc={acc_tta:.4f}')
    else:
        acc_tta=None

    # 輸出 confusion matrix（no-TTA）
    cm = dump_confusion(model, te_loader, 7, dev)
    cm_csv = Path(args.out_dir)/f'cm_{args.arch}_head8_noTTA.csv'
    with open(cm_csv,'w',newline='') as f:
        wr=csv.writer(f)
        for r in cm: wr.writerow([int(v) for v in r])
    print(f'[save cm] {cm_csv}')

    # 輸出 metrics txt；overall_acc / overall_acc_tta 供後續工具讀取
    met_txt = Path(args.out_dir)/f'metrics_{args.arch}_head8_noTTA.txt'
    with open(met_txt,'w') as f:
        f.write(f'overall_acc={acc_notta:.4f}\n')
        if acc_tta is not None:
            f.write(f'overall_acc_tta={acc_tta:.4f}\n')
    print(f'[save metrics] {met_txt}')

if __name__=='__main__':
    main()