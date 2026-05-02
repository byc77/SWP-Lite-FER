# METHOD_TO_CODE

## Model Architecture
- Backbone + SWP main model: `networks/SWP_cbpbn.py`
- ECA module: `networks/eca.py`
- GCG module: `networks/gcg.py`

## Loss Functions
- PALS loss: `losses/pals.py`
- AFG loss: `losses/focal_afg.py`

## Inference Utilities
- Precise-BN utility: `utils/precise_bn.py`

## Main Scripts
- Main fine-tuning script: `finetune_min.py`
- Main evaluation script: `eval_tta.py`
- Backbone comparison script: `cmp_backbone_head8.py`
- Confusion matrix export: `eval_save_cm_no_tta.py`
- Confusion matrix visualization: `draw_confusion_matrix_soft.py`

## Method Mapping
- Multi-stage backbone feature extraction: `networks/SWP_cbpbn.py`
- Local enhancement (ECA): `networks/eca.py`
- Global context gate (GCG): `networks/gcg.py`
- Stage-weighted pooling (SWP): `networks/SWP_cbpbn.py`
- Confusion-aware smoothing (PALS): `losses/pals.py`
- Adaptive focal weighting (AFG): `losses/focal_afg.py`
- Precise-BN and alpha-blend evaluation flow: `utils/precise_bn.py` + `eval_tta.py`
- no-TTA / flip-TTA evaluation: `eval_tta.py`
