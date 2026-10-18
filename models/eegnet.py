################################################################################
# EEGNet Model with Temperature Scaling
################################################################################

import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as ras
from models.temperature_scaler import TemperatureScaler
import copy

class EEGNet_Model(nn.Module):
    # Initialize the model with:
    # - n_chans: number of EEG channels input into the model
    # - n_times: the number of time points in the data (n_times / frequency = duration of input tensor in seconds)
    # - n_classes: number of output classes
    # - best_eval: how the best state of the model is decided during training; "loss", "accuracy", "auroc", "ece"
    def __init__(self, n_chans, n_times, n_classes = 2, device = None, best_eval = "loss"):
        np.random.seed(50)
        torch.manual_seed(50)
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_eval = best_eval.lower()

        # EEGNet model
        self.model = EEGNet(
            n_chans = n_chans,
            n_times = n_times,
            n_outputs = n_classes,
        ).to(self.device)

        self.scaler = None

    def _calculate_ece(self, probs, labels, n_bins = 10):
        if isinstance(n_bins, torch.Tensor):
            n_bins = int(n_bins.item())
        else:
            n_bins = int(n_bins)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_samples = len(probs)

        for i in range(n_bins):
            if i == n_bins - 1:
                in_bin = (probs >= bins[i]) & (probs <= bins[i+1])
            else:
                in_bin = (probs >= bins[i]) & (probs < bins[i+1])

            count = np.sum(in_bin)
            if count > 0:
                bin_acc = np.mean(labels[in_bin] == (probs[in_bin] > 0.5).astype(int))
                bin_conf = np.mean(np.maximum(probs[in_bin], 1 - probs[in_bin]))
                ece += (count / total_samples) * np.abs(bin_acc - bin_conf)

        return ece

    # Normalize data
    def _normalize(self, X):
        mean = X.mean(axis = -1, keepdims = True)
        std = X.std(axis = -1, keepdims = True) + 1e-6
        return (X - mean) / std        

    # Train the model with given data and labels
    def fit(self, X, y, batch_size = 32, lr = 1e-3, n_epochs = 40, save_dir = None, name = "eegnet model"):
        # Split for calibration
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y
        )

        X_train = np.array([self._normalize(x) for x in X_train])
        X_val = np.array([self._normalize(x) for x in X_val])

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = 1e-4)
        criterion = nn.CrossEntropyLoss()

        if self.best_eval in ["loss", "ece"]:
            best_score = float("inf")
        else:
            best_score = -float("inf")

        best_state = None

        metric_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_auroc": [],
            "val_ece": []
        }
        best_epoch = 0

        # Train model
        for epoch in range(n_epochs):
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_logits_list = []
            val_labels_list = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)

                    val_logits_list.append(logits.cpu())
                    val_labels_list.append(yb.cpu())

            val_loss /= len(val_loader.dataset)
            all_logits = torch.cat(val_logits_list)
            all_labels = torch.cat(val_labels_list).numpy()

            all_probs = torch.softmax(all_logits, dim=1)[:, 1].numpy()
            all_preds = (all_probs > 0.5).astype(int)

            val_accuracy = np.mean(all_preds == all_labels)
            try:
                val_auroc = ras(all_labels, all_probs)
            except ValueError:
                val_auroc = 0.5
            val_ece = self._calculate_ece(all_probs, all_labels)

            metric_history["train_loss"].append(train_loss / len(train_loader.dataset))
            metric_history["val_loss"].append(val_loss)
            metric_history["val_accuracy"].append(val_accuracy)
            metric_history["val_auroc"].append(val_auroc)
            metric_history["val_ece"].append(val_ece)

            if self.best_eval == "loss":
                current_score = val_loss
                improved = current_score < best_score
            elif self.best_eval == "accuracy":
                current_score = np.mean(all_preds == all_labels)
                improved = current_score > best_score
            elif self.best_eval == "auroc":
                try:
                    current_score = ras(all_labels, all_probs)
                except ValueError:
                    current_score = 0.5
                improved = current_score > best_score
            elif self.best_eval == "ece":
                current_score = self._calculate_ece(all_probs, all_labels)
                improved = current_score < best_score
            else:
                raise ValueError(f"Unknown best_eval criterion: {self.best_eval}")

            if improved:
                best_score = current_score
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1

        # Load best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logits_list = []
        labels_list = []

        self.model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                logits_list.append(logits.cpu())
                labels_list.append(yb.cpu())

        logits_val = torch.cat(logits_list).to(self.device)
        labels_val = torch.cat(labels_list).to(self.device)

        # Fit temperature scaler
        self.scaler = TemperatureScaler().to(self.device)
        self.scaler.fit(logits_val, labels_val)

        if save_dir is not None:
            epochs = range(1, len(metric_history["train_loss"]) + 1)
            fig, axes = plt.subplots(2, 2, figsize = (12, 10))

            # Plot loss
            axes[0, 0].plot(epochs, metric_history["train_loss"], label = "Train Loss")
            axes[0, 0].plot(epochs, metric_history["val_loss"], label = "Validation Loss")
            axes[0, 0].axvline(best_epoch, color = "red", linestyle = "--", label = "Selected Epoch")
            axes[0, 0].set_title("Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()

            # Plot accuracy
            axes[0, 1].plot(epochs, metric_history["val_accuracy"], label = "Validation Accuracy")
            axes[0, 1].axvline(best_epoch, color = "red", linestyle = "--", label = "Selected Epoch")
            axes[0, 1].set_title("Validation Accuracy")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()

            # Plot AUROC
            axes[1, 0].plot(epochs, metric_history["val_auroc"], label = "Validation AUROC")
            axes[1, 0].axvline(best_epoch, color = "red", linestyle = "--", label = "Selected Epoch")
            axes[1, 0].set_title("Validation AUROC")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("AUROC")
            axes[1, 0].legend()

            # Plot ECE
            axes[1, 1].plot(epochs, metric_history["val_ece"], label = "Validation ECE")
            axes[1, 1].axvline(best_epoch, color = "red", linestyle = "--", label = "Selected Epoch")
            axes[1, 1].set_title("Validation ECE")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("ECE")
            axes[1, 1].legend()

            plt.suptitle(f"{name} Training Metrics [{self.best_eval.upper()}]")
            plt.tight_layout()

            # Save training plot
            os.makedirs(save_dir, exist_ok = True)
            safe_name = name.replace(" ", "_").lower()
            plt.savefig(os.path.join(save_dir, f"{safe_name}_training_metrics.png"))
            plt.close()

    # Predict probabilities for given data
    def predict_proba(self, epoch_data):
        self.model.eval()
        x = self._normalize(epoch_data)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)

            # Temperature scaling
            if self.scaler is not None:
                logits = self.scaler(logits)

            # Softmax
            probs = torch.softmax(logits, dim = 1)
        return float(probs[0, 1].item())

    # Save the model and scaler
    def save(self, path):
        torch.save({
            "model_state": self.model.state_dict(),
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
            "n_chans": self.model.n_chans,
            "n_times": self.model.n_times,
            "device": self.device,
            "n_classes": self.model.n_outputs,
            "best_eval": self.best_eval,
        }, path)

    # Load a saved model and scaler
    @staticmethod
    def load(path, device=None):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        model = EEGNet_Model(
            n_chans=checkpoint["n_chans"],
            n_times=checkpoint["n_times"],
            n_classes=checkpoint["n_classes"],
            device=device or checkpoint["device"],
            best_eval=checkpoint.get("best_eval", "loss")
        )

        model.model.load_state_dict(checkpoint["model_state"])

        scaler_device = torch.device(device or checkpoint["device"])
        scaler = TemperatureScaler().to(scaler_device)

        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])

        model.scaler = scaler
        return model
