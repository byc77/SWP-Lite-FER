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
    使用指定資料重新估計模型中 BatchNorm 的 running_mean / running_var。
    這個步驟不更新模型權重，只讓模型前向通過若干 batch，
    以重新校正 BN 統計量，使推論時的特徵分布更穩定。
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
    將目前 BN 的校正後統計量（calib）與 snapshot 保存的訓練期統計量（train）
    依照 alpha 做線性混合。
    mean* = alpha * mean_train + (1 - alpha) * mean_calib
    var*  = alpha * var_train  + (1 - alpha) * var_calib
    """
    for (m, mean_train, var_train) in snaps:
        mean_calib = m.running_mean
        var_calib  = m.running_var
        m.running_mean.copy_(alpha * mean_train + (1.0 - alpha) * mean_calib)
        m.running_var.copy_(alpha * var_train  + (1.0 - alpha) * var_calib)