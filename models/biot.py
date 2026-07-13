################################################################################
# BIOT Model with Temperature Scaling
################################################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from braindecode.models import BIOT
from models.temperature_scaler import TemperatureScaler

class BIOT_Model(nn.Module):
    def __init__(self, n_chans = 22, n_times = 256, n_classes = 2, device = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BIOT(
            n_chans = n_chans,
            n_times = n_times,
            n_outputs = n_classes
        ).to(self.device)

    def forward(self, x):
        return self.model(x)
    
    def fit(self, X, y, batch_size = 32, lr = 1e-3, n_epochs = 40):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y
        )

        train_data = TensorDataset(
            torch.tensor(X_train, dtype = torch.float32),
            torch.tensor(y_train, dtype = torch.long)
        )
        val_data = TensorDataset(
            torch.tensor(X_val, dtype = torch.float32),
            torch.tensor(y_val, dtype = torch.long)
        )

        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = DataLoader(val_data, batch_size = batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()

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
                xb = xb.to(self.device)
                logits = self.model(xb)
                logits_list.append(logits.cpu())
                labels_list.append(yb)

        logits_val = torch.cat(logits_list)
        labels_val = torch.cat(labels_list)

        self.scaler = TemperatureScaler().to(self.device)
        self.scaler.fit(logits_val, labels_val)
    
    def predict_logits(self, epoch_data):
        x = torch.tensor(epoch_data, dtype = torch.float32).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            return self.forward(x)[0]
        
    def predict_proba(self, epoch_data):
        logits = self.predict_logits(epoch_data)
        probs = torch.softmax(logits, dim = 0)
        return float(probs[1].item())
    
    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "n_chans": self.model.n_chans,
            "n_times": self.model.n_times,
            "n_classes": self.model.n_outputs,
            "device": self.device
        }, path)

    @staticmethod
    def load(path, device = None):
        checkpoint = torch.load(path, map_location = torch.device("cpu"))

        model = BIOT_Model(
            n_chans = checkpoint["n_chans"],
            n_times = checkpoint["n_times"],
            n_classes = checkpoint["n_classes"],
            device = device or checkpoint["device"]
        )

        model.model.load_state_dict(checkpoint["model_state"])

        scaler_device = torch.device(device or checkpoint["device"])
        scaler = TemperatureScaler().to(scaler_device)
        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        model.scaler = scaler

        return model