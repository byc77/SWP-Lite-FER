import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# --- ECA（通道注意力）---
class ECA(nn.Module):
    def __init__(self, k_size: int = 3):
        super().__init__()
        assert k_size % 2 == 1 and k_size >= 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)      # (B, 1, C)
        y = self.conv(y)                         # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)

# --- FFA（簡化版 spatial attention，概念接近 CBAM 的 spatial branch）---
class FFA(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

# --- Global Context Gate (GCG)：用 s3 的全域語意去 gate s4 ---
class GlobalContextGate(nn.Module):
    """
    使用 s3 的全域語意資訊來引導並 gate s4，形成 local + global 的特徵調節。
    採用 lazy init：在第一次 forward 時，依照輸入通道數動態建立小型 MLP。
    這裡的 MLP 以 1x1 conv 實作，保持輕量化。
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

    def forward(self, s3: torch.Tensor, s4: torch.Tensor) -> torch.Tensor:
        if self.mlp is None:
            self._build(s3.size(1), s4.size(1))
        # 若 mlp 建立時所在裝置和目前輸入不同，則同步搬到相同的 CPU/GPU 裝置
        if next(self.mlp.parameters()).device != s4.device:
            self.mlp = self.mlp.to(s4.device)

        g = self.pool(s3)     # (B, Cin3, 1, 1)
        g = self.mlp(g)       # (B, Cout4, 1, 1)
        gate = self.sigmoid(g)
        return s4 * gate

# --- SWF（簡化版 Stage-Weighted Fusion）---
class SimpleSWF(nn.Module):
    """
    簡化版 SWF：將 s3 / s4 先做 GAP，再投影到同一特徵維度，
    之後透過 learnable 的 alpha 做加權融合。
    這樣可以保留較高層與次高層的資訊，並讓模型自動學習兩者權重。
    """
    def __init__(self, c3: int, c4: int, feat_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj3 = nn.Linear(c3, feat_dim, bias=False)
        self.proj4 = nn.Linear(c4, feat_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # s4 的可學習權重

    def forward(self, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        z3 = self.pool(f3).flatten(1)  # (B, c3)
        z4 = self.pool(f4).flatten(1)  # (B, c4)
        z3 = self.proj3(z3)            # (B, D)
        z4 = self.proj4(z4)            # (B, D)
        a = torch.clamp(self.alpha, 0.0, 1.0)
        out = (1.0 - a) * z3 + a * z4
        return out  # (B, D)

# --- 主模型：SWP-Lite++ with GCG ---
class SWPNet(nn.Module):
    """
    Backbone: ResNet18
    流程：backbone -> {s3, s4} -> FFA -> (ECA 可選) -> (GCG 可選) -> SWF -> Classifier
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True, feat_dim: int = 256,
                 use_eca: bool = True, use_gcg: bool = True):
        super().__init__()
        self.use_eca = bool(use_eca)
        self.use_gcg = bool(use_gcg)

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        b = resnet18(weights=weights)

        # backbone 到 layer2 為止
        self.stem = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2)
        self.layer3 = b.layer3  # s3: 256 ch
        self.layer4 = b.layer4  # s4: 512 ch

        # 局部特徵增強
        self.ffa3 = FFA(kernel_size=7)
        self.ffa4 = FFA(kernel_size=7)

        if self.use_eca:
            self.eca_s3 = ECA(k_size=3)
            self.eca_s4 = ECA(k_size=3)
        else:
            self.eca_s3 = nn.Identity()
            self.eca_s4 = nn.Identity()

        # 全域語意門控
        self.gcg = GlobalContextGate(bottleneck_ratio=16) if self.use_gcg else None

        # 融合與分類
        c3 = 256
        c4 = 512
        self.swf = SimpleSWF(c3=c3, c4=c4, feat_dim=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

        # 分類頭初始化
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='linear')
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        f3 = self.layer3(x)    # (B, 256, H/8,  W/8)
        f4 = self.layer4(f3)   # (B, 512, H/16, W/16)

        # FFA（spatial） + ECA（channel）逐層增強
        f3 = self.ffa3(f3)
        f3 = self.eca_s3(f3)
        f4 = self.ffa4(f4)
        f4 = self.eca_s4(f4)

        # GCG：用 s3 的全域語意去引導並調整 s4
        if self.gcg is not None:
            f4 = self.gcg(f3, f4)

        # SWF 融合 + 分類
        z = self.swf(f3, f4)           # (B, D)
        logits = self.classifier(z)     # (B, num_classes)
        return logits