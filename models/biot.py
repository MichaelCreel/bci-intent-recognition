################################################################################
# BIOT Model with Temperature Scaling
################################################################################

import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from braindecode.models import InterpolatedBIOT as BIOT
from models.temperature_scaler import TemperatureScaler
import copy

class BIOT_Model(nn.Module):
    # Initialize the model with:
    # - n_chans: number of EEG channels input into the model
    # - n_times: the number of time points in the data (n_times / frequency = duration of input tensor in seconds)
    # - n_classes: number of output classes
    # - frequency: sampling frequency of the data
    # - version: load a "pretrained" or "None" model
    def __init__(self, ch_names = None, n_chans = 22, n_times = 256, n_classes = 2, device = None, frequency = 250, version = "None"):
        np.random.seed(50)
        torch.manual_seed(50)
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None

        # Save original data information
        self.orig_freq = frequency
        self.orig_n_chans = n_chans

        self.target_freq = 200
        self.target_chans = 18

        # Calculate targeted n_times
        self.target_n_times = int(round(n_times * (self.target_freq / self.orig_freq)))

        if ch_names is None:
            ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

        # Save channel names to model
        self.ch_names = ch_names

        # Build channel info
        info = mne.create_info(ch_names = ch_names, sfreq = self.target_freq, ch_types = 'eeg')
        info.set_montage('standard_1020')

        self.model = BIOT(
            chs_info=info['chs'],
            n_times = self.target_n_times,
            n_outputs = n_classes,
            sfreq= self.target_freq,
        ).to(self.device)

        self.pretrained = False
        if version == "pretrained":
            self.pretrained = True
            state = torch.load("models/EEG-six-datasets-18-channels.ckpt", map_location="cpu")

            raw_state = state
            cleaned_state = {k.replace("model.", ""): v for k, v in raw_state.items()}

            self.model.load_state_dict(cleaned_state, strict=False)

    def _resample_data(self, X):
        X = self._normalize(X)
        return X

    # Forward pass through the model
    def forward(self, x):
        return self.model(x)

    # Normalize data
    def _normalize(self, X):
        mean = X.mean(axis = -1, keepdims = True)
        std = X.std(axis = -1, keepdims = True) + 1e-6
        return (X - mean) / std

    # Train the model with given data and labels
    # If pretrained model is used, training for the model is skipped while the temperature scaler is trained
    def fit(self, X, y, batch_size = 32, lr = 1e-3, n_epochs = 40):
        if self.pretrained:
            lr = lr / 100
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "head" not in name:
                    param.requires_grad = False

        #X = np.array([self._resample_data(x) for x in X_in])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y
        )

        X_train = np.array([self._resample_data(x) for x in X_train])
        X_val = np.array([self._resample_data(x) for x in X_val])

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = 1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(n_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # VALIDATION
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

            val_loss /= len(val_loader.dataset)
            val_acc = correct / total

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Train temperature scaler regardless of model training
        logits_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                logits_list.append(logits.cpu())
                labels_list.append(yb.to(self.device))

        logits_val = torch.cat(logits_list).to(self.device)
        labels_val = torch.cat(labels_list).to(self.device)

        self.scaler = TemperatureScaler().to(self.device)
        self.scaler.fit(logits_val, labels_val)
    
    # Predict logits for given data
    def predict_logits(self, epoch_data):
        x = self._resample_data(epoch_data)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            return self.forward(x)[0]

    # Predict probabilities for given data
    def predict_proba(self, epoch_data):
        x = self._resample_data(epoch_data)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

            if self.scaler is not None:
                logits = self.scaler(logits)

            probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].item())

    # Save the model and scaler
    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "ch_names": self.ch_names,
            "n_chans": len(self.ch_names) if hasattr(self.model, 'ch_names') and self.ch_names else getattr(self, "orig_n_chans", 18),
            "n_times": self.model.n_times,
            "n_classes": self.model.n_outputs,
            "frequency": getattr(self, "orig_freq", 250),
            "version": "pretrained" if self.pretrained else "None",
            "device": self.device
        }, path)

    # Load a saved model and scaler
    @staticmethod
    def load(path, device = None):
        checkpoint = torch.load(path, map_location = torch.device("cpu"))

        model = BIOT_Model(
            ch_names = checkpoint.get("ch_names", None),
            n_chans = checkpoint.get("n_chans", 18),
            n_times = checkpoint["n_times"],
            n_classes = checkpoint["n_classes"],
            frequency = checkpoint.get("frequency", 250),
            version = checkpoint.get("version", "None"),
            device = device or checkpoint["device"]
        )

        model.model.load_state_dict(checkpoint["model_state"])

        scaler_device = torch.device(device or checkpoint["device"])
        scaler = TemperatureScaler().to(scaler_device)
        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        model.scaler = scaler

        return model