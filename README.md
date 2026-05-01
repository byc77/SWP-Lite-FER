# SWP-Lite++: ICCE-TW Conference Version

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

## Dataset
This conference version uses **RAF-DB** for the main reported results.

Please prepare the dataset at:
```text
D:\SWP-Stages-Weighted-Pooling-CNN-with-FER-main\Datasets\RAF-DB
```

Required RAF-DB-related files include:
- `EmoLabel/list_patition_label.txt`
- aligned RAF-DB images arranged in the expected folder structure

Note:
- The dataset itself is **not included** in this repository.
- Please obtain RAF-DB from the official source and place it in the path above before running training or evaluation.

---

## Training
Basic fine-tuning command:
```bash
python finetune_min.py
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

To continue training, please modify the script settings or reuse the same checkpoint/output setting as needed.

---

## Testing / Inference
Main evaluation script:
```bash
python eval_tta.py
```

### SWP final: no-TTA
```bash
python eval_tta.py ^
  --raf_path "D:\SWP-Stages-Weighted-Pooling-CNN-with-FER-main\Datasets\RAF-DB" ^
  --checkpoint "D:\SWP_ICCETW_conference\checkpoints\best_rafdb_frozen.pth" ^
  --img_size 112 ^
  --batch_size 12 ^
  --workers 0 ^
  --tta_flip 0 ^
  --tta_fivecrop 0 ^
  --precise_bn_batches 800 ^
  --alpha 0.75
```

### SWP final: flip-TTA
```bash
python eval_tta.py ^
  --raf_path "D:\SWP-Stages-Weighted-Pooling-CNN-with-FER-main\Datasets\RAF-DB" ^
  --checkpoint "D:\SWP_ICCETW_conference\checkpoints\best_rafdb_frozen.pth" ^
  --img_size 112 ^
  --batch_size 12 ^
  --workers 0 ^
  --tta_flip 1 ^
  --tta_fivecrop 0 ^
  --precise_bn_batches 800 ^
  --alpha 0.75
```

### ResNet18 baseline: no-TTA
```bash
python eval_tta.py ^
  --raf_path "D:\SWP-Stages-Weighted-Pooling-CNN-with-FER-main\Datasets\RAF-DB" ^
  --checkpoint "D:\SWP_ICCETW_conference\checkpoints\cmp_head8_base.pth" ^
  --img_size 112 ^
  --batch_size 12 ^
  --workers 0 ^
  --tta_flip 0 ^
  --tta_fivecrop 0 ^
  --precise_bn_batches 800 ^
  --alpha 0.75
```

### ResNet18 baseline: flip-TTA
```bash
python eval_tta.py ^
  --raf_path "D:\SWP-Stages-Weighted-Pooling-CNN-with-FER-main\Datasets\RAF-DB" ^
  --checkpoint "D:\SWP_ICCETW_conference\checkpoints\cmp_head8_base.pth" ^
  --img_size 112 ^
  --batch_size 12 ^
  --workers 0 ^
  --tta_flip 1 ^
  --tta_fivecrop 0 ^
  --precise_bn_batches 800 ^
  --alpha 0.75
```

---

## Expected Results
Approximate rechecked results under the current conference folder setup:

- SWP final no-TTA: ~0.8387
- SWP final flip-TTA: ~0.8504
- ResNet18 baseline no-TTA: ~0.8344
- ResNet18 baseline flip-TTA: ~0.8432

Small numerical differences may occur across environments, but the reproduced results should remain very close to the reported conference results.

---

## Repository Structure
```text
SWP_ICCETW_conference/
├─ README.md
├─ requirements.txt
├─ train_help.txt
├─ finetune_min.py
├─ eval_tta.py
├─ cmp_backbone_head8.py
├─ eval_save_cm_no_tta.py
├─ draw_confusion_matrix_soft.py
├─ networks/
│  ├─ SWP_cbpbn.py
│  ├─ eca.py
│  └─ gcg.py
├─ losses/
│  ├─ pals.py
│  └─ focal_afg.py
├─ utils/
│  └─ precise_bn.py
├─ tools/
│  ├─ confusion_to_report.py
│  └─ count_params_flops.py
├─ checkpoints/
└─ runs/
```

---

## Notes
- `no-TTA` means single-pass inference on the original image only.
- `flip-TTA` means performing inference on both the original image and its horizontally flipped version, then averaging the outputs.
- `Precise-BN` and `alpha-blend (alpha = 0.75)` are used during evaluation for more stable inference.
- This repository is prepared for the ICCE-TW conference version only.

---

## License
This repository is prepared for academic research and reproducibility purposes only.  
The RAF-DB dataset is not redistributed in this repository.  
Please check dataset licenses and usage restrictions separately before public release.
