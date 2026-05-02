# RESULTS_MAP

## Main Conference Results

### 1. SWP Final Model (no-TTA)
- Script: `eval_tta.py`
- Checkpoint: `checkpoints/best_rafdb_frozen.pth`
- Dataset: RAF-DB
- Key settings:
  - `--tta_flip 0`
  - `--precise_bn_batches 800`
  - `--alpha 0.75`
- Expected result:
  - test acc ≈ 0.8387

### 2. SWP Final Model (flip-TTA)
- Script: `eval_tta.py`
- Checkpoint: `checkpoints/best_rafdb_frozen.pth`
- Dataset: RAF-DB
- Key settings:
  - `--tta_flip 1`
  - `--precise_bn_batches 800`
  - `--alpha 0.75`
- Expected result:
  - test acc ≈ 0.8504

### 3. ResNet18 Baseline (no-TTA)
- Script: `eval_tta.py`
- Checkpoint: `checkpoints/cmp_head8_base.pth`
- Dataset: RAF-DB
- Key settings:
  - `--tta_flip 0`
  - `--precise_bn_batches 800`
  - `--alpha 0.75`
- Expected result:
  - test acc ≈ 0.8344

### 4. ResNet18 Baseline (flip-TTA)
- Script: `eval_tta.py`
- Checkpoint: `checkpoints/cmp_head8_base.pth`
- Dataset: RAF-DB
- Key settings:
  - `--tta_flip 1`
  - `--precise_bn_batches 800`
  - `--alpha 0.75`
- Expected result:
  - test acc ≈ 0.8432

## Notes
- `no-TTA` means single-pass inference on the original image.
- `flip-TTA` means averaging predictions from the original image and its horizontally flipped version.
- Small numerical differences may occur across environments, but results should remain close to the values above.
