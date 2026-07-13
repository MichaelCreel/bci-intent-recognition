################################################################################
# EEGNet Model with Temperature Scaling
################################################################################

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from sklearn.model_selection import train_test_split
from models.temperature_scaler import TemperatureScaler
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

        logits_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                logits_list.append(logits.cpu())
                labels_list.append(yb.cpu())

        logits_val = torch.cat(logits_list)
        labels_val = torch.cat(labels_list)

        # Fit temperature scaler
        self.scaler = TemperatureScaler().to(self.device)
        self.scaler.fit(logits_val, labels_val)

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
            "device": self.device,
            "n_classes": self.model.n_outputs,
        }, path)

    @staticmethod
    def load(path, device=None):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        model = EEGNet_Model(
            n_chans=checkpoint["n_chans"],
            n_times=checkpoint["n_times"],
            n_classes=checkpoint["n_classes"],
            device=device or checkpoint["device"]
        )

        model.model.load_state_dict(checkpoint["model_state"])

        scaler_device = torch.device(device or checkpoint["device"])
        scaler = TemperatureScaler().to(scaler_device)

        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        model.scaler = scaler
        return model
