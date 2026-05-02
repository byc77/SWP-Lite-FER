# utils/precise_bn.py
import torch
import torch.nn as nn

@torch.no_grad()
def snapshot_bn_stats(model: nn.Module):
    snaps = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            snaps.append((m, m.running_mean.clone(), m.running_var.clone()))
    return snaps

@torch.no_grad()
def precise_bn_update(model: nn.Module, data_loader, num_batches: int = 200, device: str = 'cuda'):
    """
    使用資料流重新估計 BN 統計（running_mean/var）。
    只前向，不做反傳；建議用訓練集的無標籤流。
    """
    was_training = model.training
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    it = iter(data_loader)
    for _ in range(num_batches):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        images = images.to(device, non_blocking=True)
        _ = model(images)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(was_training)

@torch.no_grad()
def blend_bn_from_snapshot(model: nn.Module, snaps, alpha: float):
    """
    當前 BN 視為 calib，snapshot 視為 train；以 alpha 混合覆寫：
    mean* = α mean_train + (1-α) mean_calib；var 同理。
    """
    for (m, mean_train, var_train) in snaps:
        mean_calib = m.running_mean
        var_calib  = m.running_var
        m.running_mean.copy_(alpha * mean_train + (1.0 - alpha) * mean_calib)
        m.running_var.copy_(alpha * var_train  + (1.0 - alpha) * var_calib)
