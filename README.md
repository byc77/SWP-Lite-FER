# SWP-Lite-FER

## Environment
- Python: 3.10.20
- PyTorch: 2.5.1+cu124
- Torchvision: 0.20.1+cu124
- CUDA: 12.4
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- OS: Windows 11
- Evaluation setting: GPU evaluation is supported; CPU-only recheck is also supported.

Install the required packages:
```bash
pip install -r requirements.txt
```
---

## Training
Basic training command:
```bash
python train.py --name {experiment_name} --dataset {dataset_name} --data_root {dataset_path} --num_classes {num_classes} --epochs {epochs} --batch_size {batch_size} --lr {learning_rate} --gpu_ids 0 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1
```

RAF-DB
```bash
python train.py --name rafdb_swp_litepp --dataset rafdb --data_root "Datasets/RAF-DB" --num_classes 7 --epochs 8 --batch_size 64 --lr 8e-5 --gpu_ids 0 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1
```

FERPlus
```bash
python train.py --name ferplus_swp_litepp --dataset ferplus --data_root "Datasets/FERPlus" --num_classes 8 --epochs 8 --batch_size 64 --lr 8e-5 --gpu_ids 0 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1
```

AffectNet
```bash
python train.py --name affectnet_swp_litepp --dataset affectnet --data_root "Datasets/AffectNet_ImageFolder" --num_classes 8 --epochs 8 --batch_size 64 --lr 8e-5 --gpu_ids 0 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1
```

Emo135
```bash
python train.py --name emo135_swp_litepp --dataset emo135 --data_root "Datasets/Emo135" --num_classes 135 --epochs 8 --batch_size 64 --lr 8e-5 --gpu_ids 0 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1
```
---

## Testing / Inference
RAF-DB no-TTA
```bash
python eval_tta.py --raf_path "Datasets/RAF-DB" --checkpoint "checkpoints/best_rafdb_frozen.pth" --img_size 112 --batch_size 12 --workers 0 --tta_flip 0 --tta_fivecrop 0 --precise_bn_batches 800 --alpha 0.75 --save_confmat "runs/rafdb/cm_best_checkpoint_noTTA.csv"
```
RAF-DB flip-TTA
```bash
python eval_tta.py --raf_path "Datasets/RAF-DB" --checkpoint "checkpoints/best_rafdb_frozen.pth" --img_size 112 --batch_size 12 --workers 0 --tta_flip 1 --tta_fivecrop 0 --precise_bn_batches 800 --alpha 0.75 --save_confmat "runs/rafdb/cm_best_checkpoint_TTA.csv"
```
---

## Checkpoints
The checkpoint files are not included in this repository due to file size limits.

Please download the required checkpoints from the following Google Drive folder and place them under:
```text
checkpoints/
```

Google Drive:
- https://drive.google.com/drive/folders/1AZ8xEU1li87C68uFwHVv2lXZH1tuhLG4?usp=drive_link
---
## Experimental Settings
- Backbone: ResNet-18
- Input size: 112 × 112
- Epochs: 8
- Learning rate: 8e-5
- Batch size: 64
- Optimizer: AdamW
- Main modules: ECA, GCG, PALS, AFG
- Inference: Precise-BN, alpha-blend, flip-TTA

---

## Notes
- `train.py` is used for standard training reproduction.
- `eval_tta.py` is used for inference reproduction with a provided checkpoint.
- `finetune_min.py` is retained for legacy fine-tuning experiments.

---

## License
This repository is prepared for academic research and reproducibility purposes only.  
Datasets are not redistributed in this repository.  
Please check dataset licenses and usage restrictions separately before public release.

