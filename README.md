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
python train.py --name rafdb_full --dataset rafdb --data_root "Datasets/RAF-DB" --num_classes 7 --epochs 200 --batch_size 128 --lr 8e-5 --workers 4 --use_ffa 1 --use_swf 1 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1 --flip_cons 1
```

FERPlus
```bash
python train.py --name ferplus_full --dataset ferplus --data_root "Datasets/FERPlus" --num_classes 8 --epochs 30 --batch_size 128 --lr 5e-6 --weight_decay 2e-3 --workers 4 --use_ffa 1 --use_swf 1 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1 --flip_cons 1
```
---

## Testing / Inference
Calibration Evaluation
```bash
python ts_calibrate_eval_clean.py --dataset rafdb --data_root "Datasets/RAF-DB" --checkpoint "checkpoints/rafdb_full/best.pth" --num_classes 7 --batch_size 128 --workers 4 --gpu_ids 0 --precise_bn_batches 800 --alpha 0.75 --out_txt "runs/rafdb_full/calibration_ts.txt"
```
CPU Deployment Profiling
```bash
python profile_swp_litepp.py --checkpoint "checkpoints/rafdb_full/best.pth" --device cpu --img_size 112 --warmup 50 --repeat 300 --num_threads 12 --out_txt "runs/rafdb_full/profile_cpu.txt"
```
---

## Checkpoints
The checkpoint files are not included in this repository due to file size limits.

Please download the required checkpoints from the following Google Drive folder and place them under:
```text
checkpoints/rafdb_full/best.pth
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

## License
This repository is prepared for academic research and reproducibility purposes only.  
Datasets are not redistributed in this repository.  
Please check dataset licenses and usage restrictions separately before public release.

