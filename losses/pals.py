# losses/pals.py
import torch

def build_pals_soft_targets(targets: torch.Tensor,
                            num_classes: int = 7,
                            eps: float = 0.10,
                            device=None) -> torch.Tensor:
    """
    PALS: Pair-Aware Label Smoothing，用於建立符合易混淆類別關係的 soft targets。
    這裡以 RAF-DB 的 7 類為例，類別索引為：
    [0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger]
    透過事先定義的易混淆類別配對，將平滑機率 eps 分配到鄰近類別，
    而不是平均分配到所有非真實類別。
    """
    PAIRS = {
        5: [6, 2],  # Disgust -> Anger, Sadness
        4: [3],     # Fear    -> Surprise
        6: [5],     # Anger   -> Disgust（對稱配對）
        3: [4],     # Surprise-> Fear    （對稱配對）
    }
    targets = targets.long()
    if device is None:
        device = targets.device
    B = targets.size(0)
    y = torch.zeros(B, num_classes, device=device, dtype=torch.float32)
    y.scatter_(1, targets.view(-1, 1), 1.0 - eps)

    for i, t in enumerate(targets.tolist()):
        if t in PAIRS and len(PAIRS[t]) > 0:
            share = eps / len(PAIRS[t])
            for n in PAIRS[t]:
                y[i, n] += share
    return y