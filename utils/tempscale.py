# utils/tempscale.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelWithTemperature(nn.Module):
    """
    Temperature scaling（校準機率，部署友善）。
    推論時用 logits / T；不改準確率但改善 ECE/Brier。
    """
    def __init__(self, model: nn.Module, T_init: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * T_init)

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        T = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / T

    @torch.no_grad()
    def predict_proba(self, x):
        logits = self.model(x)
        return F.softmax(logits / self.temperature, dim=1)

@torch.no_grad()
def fit_temperature(wrapper_model: ModelWithTemperature, valid_loader, device='cuda', max_iter: int = 5):
    """
    在驗證集上最小化 NLL 來擬合溫度 T。
    """
    nll_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.LBFGS([wrapper_model.temperature], lr=0.01, max_iter=50)

    def _nll():
        loss_sum = 0.0
        n = 0
        for images, targets in valid_loader:
            images = images.to(device); targets = targets.to(device)
            logits = wrapper_model.model(images)
            loss = nll_criterion(logits / wrapper_model.temperature, targets)
            loss_sum += loss.item() * images.size(0)
            n += images.size(0)
        return loss_sum / max(1, n)

    def closure():
        optimizer.zero_grad()
        loss_val = torch.tensor(_nll(), requires_grad=True)
        loss_val.backward()
        return loss_val

    for _ in range(max_iter):
        optimizer.step(closure)
    return wrapper_model
