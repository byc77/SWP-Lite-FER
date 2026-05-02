# networks/gcg.py
import torch
import torch.nn as nn

class GlobalContextGate(nn.Module):
    """
    GCG: ??s3 ?撅銝?? gate s4 ??嚗ocal?lobal ??嚗?
    Lazy init嚗洵銝甈?forward 靘撓?仿??芸?撱箏???MLP??
    ?嚗撅?1x1 conv 雿??GAP(s3) 銝?撟曆? 0 ???FLOPs??
    """
    def __init__(self, bottleneck_ratio: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck_ratio = bottleneck_ratio
        self.mlp = None
        self.sigmoid = nn.Sigmoid()

    def _build(self, cin3: int, cout4: int):
        mid = max(4, cin3 // self.bottleneck_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(cin3, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout4, kernel_size=1, bias=False)
        )
        self.mlp.to(next(self.parameters()).device)

    def forward(self, s3: torch.Tensor, s4: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            self._build(s3.size(1), s4.size(1))
        g = self.pool(s3)         # (B, Cin3, 1, 1)
        g = self.mlp(g)           # (B, Cout4, 1, 1)
        gate = self.sigmoid(g)
        return s4 * gate


