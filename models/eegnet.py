################################################################################
# EEGNet Model with Temperature Scaling
################################################################################

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from sklearn.model_selection import train_test_split

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature
    
def fit_temperature_scaler(model, val_loader, device):
    model.eval()
    scaler = TemperatureScaler().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr = 0.01, max_iter = 50)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            logits_list.append(logits)
            labels_list.append(yb)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    return scaler

class EEGNet_Model:
    def __init__(self, n_chans, n_times, n_classes = 2, device = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # EEGNet model
        self.model = EEGNet(
            n_chans = n_chans,
            n_times = n_times,
            n_outputs = n_classes,
        ).to(self.device)

        self.scaler = None
    
    def fit(self, X, y, batch_size = 32, lr = 1e-3, n_epochs = 40):
        # Split for calibration
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y
        )

        # Build dataloaders
        train_data = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype = torch.float32),
            torch.tensor(y_train, dtype = torch.long)
        )
        val_data = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype = torch.float32),
            torch.tensor(y_val, dtype = torch.long)
        )

        train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()

        # Train model
        for epoch in range(n_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Fit temperature scaler
        self.scaler = fit_temperature_scaler(self.model, val_loader, self.device)

    def predict_proba(self, epoch_data):
        x = torch.tensor(epoch_data, dtype = torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

            # Temperature scaling
            if self.scaler is not None:
                logits = self.scaler(logits)

            # Softmax
            probs = torch.softmax(logits, dim = 1)
        return float(probs[0, 1].item())

    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "n_chans": self.model.n_chans,
            "n_times": self.model.n_times,
            "device": self.device
        }, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location = torch.device("cpu"))
        model = EEGNet_Model(
            n_chans = checkpoint["n_chans"],
            n_times = checkpoint["n_times"],
            device = "cpu"
        )

        model.model.load_state_dict(checkpoint["model_state"])

        scaler = TemperatureScaler()
        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        model.scaler = scaler
        
        return model