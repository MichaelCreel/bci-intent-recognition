################################################################################
# Interpolated BIOT Model with Temperature Scaling
################################################################################

import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as ras
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
    # - best_eval: how the best state of the model is decided during training; "loss", "accuracy", "auroc", "ece"
    # - resample: whether to resample training and input data to the proper frequency for BIOT
    # - display_resample: whether to display a plot of the first channel before and after resampling
    def __init__(self, ch_names = None, n_chans = 22, n_times = 256, n_classes = 2, device = None, frequency = 250, version = "None", best_eval = "loss", resample = False, display_resample = False):
        np.random.seed(50)
        torch.manual_seed(50)
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.best_eval = best_eval.lower()
        self.resample = resample
        self.display_resample = display_resample

        # Save original data information
        self.orig_freq = frequency
        if version == "pretrained":
            self.target_freq = 200
            self.target_n_times = 256
        elif resample:
            self.target_freq = 200
            # Calculate targeted n_times
            self.target_n_times = int(round(n_times * (self.target_freq / self.orig_freq)))
        else:
            self.target_freq = frequency
            self.target_n_times = n_times

        self.orig_n_chans = n_chans
        self.target_chans = 18

        if ch_names is None:
            ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']

        # Save channel names to model
        self.ch_names = ch_names

        # Build channel info
        info = mne.create_info(ch_names = ch_names, sfreq = self.target_freq, ch_types = 'eeg')
        info.set_montage('standard_1020')

        self.pretrained = False
        if version == "pretrained":
            self.pretrained = True
            self.model = BIOT.from_pretrained(
                "braindecode/biot-pretrained-six-datasets-18chs",
                chs_info = info['chs'],
                n_outputs = n_classes,
            ).to(self.device)
        else:
            self.model = BIOT(
                chs_info=info['chs'],
                n_times = self.target_n_times,
                n_outputs = n_classes,
                sfreq = self.target_freq,
                trainable = True
            ).to(self.device)

    # Resample a single epoch to the target frequency
    def _resample_epoch(self, epoch_data):
        if not self.resample:
            return epoch_data
        info = mne.create_info(
            ch_names = self.ch_names,
            sfreq = self.orig_freq,
            ch_types = 'eeg'
        )
        epoch_data = epoch_data[np.newaxis, :, :]

        epochs = mne.EpochsArray(
            epoch_data,
            info,
            verbose = False
        )

        epochs.resample(self.target_freq)

        if self.display_resample:
            # Compare first channel before/after resampling
            original = epoch_data[0, 0, :]          # Channel 0 before resampling
            resampled = epochs.get_data()[0, 0, :]  # Channel 0 after resampling

            plt.plot(
                np.arange(len(original)) / self.orig_freq,
                original,
                label="Original",
                alpha=0.8
            )

            plt.plot(
                np.arange(len(resampled)) / self.target_freq,
                resampled,
                label="Resampled",
                alpha=0.8
            )

            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title("Channel 0 Before and After Resampling")
            plt.legend()

            plt.tight_layout()
            plt.show(block=True)

        return epochs.get_data()[0]
    
    def _calculate_ece(self, probs, labels, n_bins = 10):
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if isinstance(n_bins, torch.Tensor):
            n_bins = int(n_bins.item())
        elif hasattr(n_bins, "item"):
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

    # Forward pass through the model
    def forward(self, x):
        return self.model(x)

    # Normalize data
    def _normalize(self, X):
        mean = X.mean(axis = -1, keepdims = True)
        std = X.std(axis = -1, keepdims = True) + 1e-6
        return (X - mean) / std

    # Train the model with given data and labels
    # If pretrained model is used, the classifier head and the temperature scaler are trained
    # save_dir: directory to save the training metrics plot, unsaved if None
    # name: name of the model to be prepended to the training metrics plot
    def fit(self, X, y, batch_size = 32, lr = 1e-3, n_epochs = 40, save_dir = None, name = "biot model"):
        # Resample training data
        X = np.array([self._resample_epoch(x) for x in X])

        if self.pretrained:
            lr = lr / 100
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "head" not in name:
                    param.requires_grad = False

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y
        )

        X_train = np.array([self._normalize(x) for x in X_train])
        X_val = np.array([self._normalize(x) for x in X_val])

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

            # VALIDATION
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

            if improved:
                best_score = current_score
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1

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
    
    # Predict logits for given data
    def predict_logits(self, epoch_data):
        epoch_data = self._resample_epoch(epoch_data)
        x = self._normalize(epoch_data)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            return self.forward(x)[0]

    # Predict probabilities for given data
    def predict_proba(self, epoch_data):
        epoch_data = self._resample_epoch(epoch_data)

        self.model.eval()
        x = self._normalize(epoch_data)
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
            "best_eval": self.best_eval,
            "resample": self.resample,
            "display_resample": self.display_resample,
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
            best_eval = checkpoint.get("best_eval", "loss"),
            device = device or checkpoint["device"],
            resample = checkpoint.get("resample", True),
            display_resample = checkpoint.get("display_resample", False)
        )

        model.model.load_state_dict(checkpoint["model_state"])

        scaler_device = torch.device(device or checkpoint["device"])
        scaler = TemperatureScaler().to(scaler_device)
        if checkpoint["scaler_state"] is not None:
            scaler.load_state_dict(checkpoint["scaler_state"])
        model.scaler = scaler

        return model