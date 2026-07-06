################################################################################
# Compares BIOT Pre-trained Model with EEGNet Model
################################################################################

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from transformers import AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

# Import EEGNet from models
from models.eegnet import EEGNet_Model

def build_epochs_for_subject(subject_id):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes = 2,
        events = ["left_hand", "right_hand"],
        fmin = 8,
        fmax = 30,
    )

    X, y, meta = paradigm.get_data(dataset, subjects = [subject_id])

    # Convert string labels into integer labels
    classes, y_int = np.unique(y, return_inverse = True)

    n_channels = X.shape[1]

    montage = mne.channels.make_standard_montage("standard_1020")
    channel_names = montage.ch_names[:n_channels]

    info = mne.create_info(
        ch_names = channel_names,
        sfreq = 250,
        ch_types = "eeg"
    )
    info.set_montage(montage)

    epochs = mne.EpochsArray(X, info)
    return epochs, y_int

class BIOT_Model(nn.Module):
    def __init__(self, model_name = "neurotechlab/biot-eeg-full", n_classes = 2, device = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.to(self.device)

        for p in self.backbone.parameters():
            p.requires_grad = False

        # Determine embedding dimension
        dummy_input = torch.zeros(1, 22, 256).to(self.device)
        with torch.no_grad():
            out = self.backbone(dummy_input)
        embedding_dim = out.last_hidden_state.shape[-1]

        # Linear head embeddings to n classes
        self.head = nn.Linear(embedding_dim, n_classes).to(self.device)

    def forward(self, x):
        out = self.backbone(x)
        embeddings = out.last_hidden_state.mean(dim = 1)
        logits = self.head(embeddings)
        return logits
    
    # Train the linear head
    def fit(self, X, y, batch_size = 32, lr = 1e-3, n_epochs = 20):
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

        optimizer = torch.optim.Adam(self.head.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        self.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.forward(xb)
                preds.append(logits.argmax(dim = 1).cpu().numpy())
                labels.append(yb.cpu().numpy())

        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        print(f"Validation Accuracy: {accuracy_score(labels, preds):.4f}")

    def predict_proba(self, epoch_data):
        x = torch.tensor(epoch_data, dtype = torch.float32).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim = 1)
        return float(probs[0, 1].item())
