# losses/focal_afg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossAFG(nn.Module):
    """
    支援每個類別使用不同 gamma 的 Focal Loss，並可搭配 soft targets。
    y_soft: (B, C)，通常來自 PALS 產生的 soft labels
    logits: (B, C)
    """
    def __init__(self, class_gamma: torch.Tensor = None,
                 reduction: str = 'mean', eps: float = 1e-8):
        super().__init__()
        if class_gamma is not None and not isinstance(class_gamma, torch.Tensor):
            class_gamma = torch.tensor(class_gamma, dtype=torch.float32)
        self.register_buffer('class_gamma', class_gamma if class_gamma is not None else None)
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
        # 先將 softmax 機率限制在安全範圍內，避免 in-place 或數值過小造成 autograd / log 計算不穩定
        p = torch.clamp(F.softmax(logits, dim=1), min=self.eps, max=1.0 - self.eps)  # (B, C)
        if self.class_gamma is None:
            focal = (1 - p).pow(2.0)
        else:
            focal = (1 - p).pow(self.class_gamma)  # (C,) 會自動 broadcast 到 (B, C)
        loss = -(y_soft * focal * torch.log(p)).sum(dim=1)  # (B,)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss