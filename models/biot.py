################################################################################
# BIOT Model with Temperature Scaling
################################################################################

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from braindecode.models import BIOT

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, labels, lr = 1e-2, n_epochs = 200):
        optimizer = torch.optim.LBFGS([self.temperature], lr = lr, max_iter = n_epochs)

        logits = torch.tensor(logits, dtype = torch.float32)
        labels = torch.tensor(labels, dtype = torch.long)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)

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
    
    def predict_logits(self, epoch_data):
        x = torch.tensor(epoch_data, dtype = torch.float32).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            return self.forward(x)[0]
        
    def predict_proba(self, epoch_data):
        logits = self.predict_logits(epoch_data)
        probs = torch.softmax(logits, dim = 0)
        return float(probs[1].item())
