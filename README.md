# SWP-Lite-FER

## Environment
- Python: 3.10.19
- PyTorch: 2.8.0
- Torchvision: 0.23.0
- OS: Windows 11
- Hardware: AMD Ryzen 5 7535HS CPU, 32 GB RAM
- Evaluation setting: CPU-only recheck is supported

Make sure to install the required Python packages from `requirements.txt`.  
It is recommended to use a virtual environment or Conda environment before running the code.

---

## Training
Basic fine-tuning command:
```bash
python train.py --name {experiment_name} --dataset {dataset_name} --data_root {dataset_path} --epochs {epochs} --batch_size {batch_size} --lr {learning_rate} --gpu_ids 0 --use_eca 1 --use_gcg 1 --use_pals 1 --use_afg 1
```

Experimental Settings:
- Backbone: ResNet-18
- Input Size: 112 × 112
- Fine-tuning Epochs: 8–10
- Learning Rate: 8e-5

Main training-related modules:
- `losses/pals.py`: PALS-related implementation
- `losses/focal_afg.py`: AFG-related implementation
- `finetune_min.py`: head-only fine-tuning pipeline

Resume Training:  
To continue training, please modify the script settings or reuse the same checkpoint/output setting as needed.

---

## Testing / Inference
```bash
python eval_tta.py --raf_path {your_rafdb_path} --checkpoint {your_checkpoint_path} --img_size 112 --batch_size 12 --workers 0 --tta_flip 1 --tta_fivecrop 0 --precise_bn_batches 800 --alpha 0.75
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

## License
This repository is prepared for academic research and reproducibility purposes only.  
The RAF-DB dataset is not redistributed in this repository.  
Please check dataset licenses and usage restrictions separately before public release.

