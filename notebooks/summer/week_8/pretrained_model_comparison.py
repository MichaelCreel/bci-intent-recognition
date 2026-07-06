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
from braindecode.models import BIOT

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
    def __init__(self, model_name = "braindecode/biot", n_classes = 2, device = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.backbone = BIOT(
            n_chans = 22,
            n_times = 256,
            n_outputs = 2
        ).to(self.device)

        # Freeze backbone parameters to prevent training
        # for p in self.backbone.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)
    
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

        optimizer = torch.optim.Adam(self.backbone.parameters(), lr = lr)
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
    
def evaluate_model(model, epochs_test, y_Test, name = "Model"):
    preds = []
    confs = []

    for i in range(len(epochs_test)):
        epoch_data = epochs_test.get_data()[i]
        prob_right = model.predict_proba(epoch_data)
        confs.append(prob_right)
        preds.append(1 if prob_right > 0.5 else 0)

    preds = np.array(preds)
    y_test = np.array(y_Test)

    accuracy = accuracy_score(y_test, preds)
    mean_conf = np.mean(confs)
    std_conf = np.std(confs)

    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Confidence: {mean_conf:.4f}")
    print(f"Std Confidence: {std_conf:.4f}")

    return accuracy, mean_conf, std_conf

def main():
    training_subjects = [1, 2, 3, 5, 6, 7]
    test_subject = 4

    print("Loading training subjects...")
    train_epochs_list = []
    train_labels_list = []

    for subj in training_subjects:
        epochs, labels = build_epochs_for_subject(subj)
        train_epochs_list.append(epochs)
        train_labels_list.append(labels)

    epochs_train = mne.concatenate_epochs(train_epochs_list)
    y_train = np.concatenate(train_labels_list)

    X_train = epochs_train.get_data()
    n_chans = X_train.shape[1]
    n_times = X_train.shape[2]

    print(f"Training data shape: {X_train.shape}")

    # Train EEGNet model
    print("Training EEGNet Model...")
    eegnet_model = EEGNet_Model(n_chans = n_chans, n_times = n_times)
    eegnet_model.fit(X_train, y_train, batch_size = 32, lr = 1e-3, n_epochs = 40)

    # Train BIOT model
    print("Training BIOT Model...")
    biot_model = BIOT_Model(model_name = "braindecode/biot", n_classes = 2)
    biot_model.fit(X_train, y_train, batch_size = 32, lr = 1e-3, n_epochs = 40)

    print("Loading test subject...")
    epochs_test, y_test = build_epochs_for_subject(test_subject)

    # Evaluate EEGNet model
    accuracy_eegnet, conf_eegnet, std_eegnet = evaluate_model(
        eegnet_model, epochs_test, y_test, name = "EEGNet"
    )

    # Evaluate BIOT model
    accuracy_biot, conf_biot, std_biot = evaluate_model(
        biot_model, epochs_test, y_test, name = "BIOT"
    )

    print("\n=== Summary ===")
    print(f"EEGNet vs BIOT Accuracy: {accuracy_eegnet:.4f} vs {accuracy_biot:.4f}")
    print(f"EEGNet vs BIOT Mean Confidence: {conf_eegnet:.4f} vs {conf_biot:.4f}")
    print(f"EEGNet vs BIOT Std Confidence: {std_eegnet:.4f} vs {std_biot:.4f}")

if __name__ == "__main__":
    main()
