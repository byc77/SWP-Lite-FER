# networks/eca.py
import torch
import torch.nn as nn

class ECA(nn.Module):
    """
    Efficient Channel Attention (ultra-light).
    雿輻 1D ?瑞??????其漱鈭??∪??撅歹???扔撠?    """
    def __init__(self, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1 and k_size >= 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)      # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


