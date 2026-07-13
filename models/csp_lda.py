################################################################################
# CSP + LDA Model with Temperature Scaling
################################################################################

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from models.temperature_scaler import TemperatureScaler

class CSP_LDA_Model:
    def __init__(self, n_components = 6):
        self.n_components = n_components
        self.csp = None
        self.lda = None
        self.scaler = None

    # Training
    def fit(self, X, y):
        #Split for calibration
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y
        )

        # CSP
        self.csp = CSP(n_components = self.n_components, log = True)
        X_train_csp = self.csp.fit_transform(X_train, y_train)
        X_val_csp = self.csp.transform(X_val)

        # LDA
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_train_csp, y_train)

        # Validation logits
        logits_val = self.lda.decision_function(X_val_csp)

        # Convert to 2-class logits
        logits_val = np.column_stack([-logits_val, logits_val])

        # Fit temperature scaler
        self.scaler = TemperatureScaler()
        self.scaler.fit(logits_val, y_val)

    def predict_proba(self, epoch_data):
        X = epoch_data[np.newaxis, :, :]

        # CSP Transform
        X_csp = self.csp.transform(X)

        # LDA Logits
        logits = self.lda.decision_function(X_csp)
        logits = np.column_stack([-logits, logits])

        # Apply Temperature Scaling
        logits_t = torch.tensor(logits, dtype = torch.float32)
        scaled_logits = self.scaler(logits_t).detach().numpy()[0]

        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        return float(probs[1])
    
    def save(self, path):
        torch.save({
            "csp": self.csp,
            "lda": self.lda,
            "temperature": self.scaler.state_dict(),
            "n_components": self.n_components
        }, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location = torch.device("cpu"))
        model = CSP_LDA_Model(n_components = checkpoint["n_components"])
        model.csp = checkpoint["csp"]
        model.lda = checkpoint["lda"]

        scaler = TemperatureScaler().to(torch.device or checkpoint["device"])
        scaler.load_state_dict(checkpoint["temperature"])
        model.scaler = scaler

        return model