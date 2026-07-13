################################################################################
# Temperature Scaler for Safety Models
################################################################################

import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr=1e-2, max_iter=200):
        # Ensure logits and labels are on the SAME device as the scaler
        device = self.temperature.device

        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32, device=device)
        else:
            logits = logits.to(device)

        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()
