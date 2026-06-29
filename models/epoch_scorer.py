################################################################################
# Determines the quality of an epoch
################################################################################

import numpy as np
import torch
import mne
from autoreject import AutoReject
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Extract features
def extract_features(epoch_data, sfreq):
    features = []

    # Amplitudes
    ptp = np.ptp(epoch_data, axis = 1)
    features.append(np.mean(ptp))
    features.append(np.std(ptp))
    features.append(np.max(ptp))

    # Variances
    var = np.var(epoch_data, axis = 1)
    features.append(np.mean(var))
    features.append(np.std(var))

    # FFT
    fft = np.abs(np.fft.rfft(epoch_data, axis = 1))
    freqs = np.fft.rfftfreq(epoch_data.shape[1], 1 / sfreq)

    # High frequency
    high_freq_band = (freqs > 20) & (freqs <= 40)
    high_frequency_energy = fft[:, high_freq_band].mean(axis = 1)
    features.append(np.mean(high_frequency_energy))
    features.append(np.std(high_frequency_energy))

    # Noise
    noise_band = (freqs > 55) & (freqs <= 65)
    noise_energy = fft[:, noise_band].mean(axis = 1)
    features.append(np.mean(noise_energy))

    # Flatline
    zero_crossings = np.sum(np.diff(np.sign(epoch_data), axis = 1) != 0, axis = 1)
    features.append(np.mean(zero_crossings))

    return np.array(features, dtype = np.float32)

# Dataset builder
def build_dataset(epochs):
    sfreq = epochs.info['sfreq']

    ar = AutoReject(consensus = [0.5], n_interpolate = [1, 2, 3])
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    labels = (~reject_log.bad_epochs).astype(int)

    X = []
    for i in range(len(epochs)):
        epoch_data = epochs.get_data()[i]
        features = extract_features(epoch_data, sfreq)
        X.append(features)

    return np.vstack(X), labels

class QualityModel:
    def __init__(self):
        self.clf = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators = 200,
                max_depth = 6,
                class_weight = 'balanced',
                random_state = 42
            ))
        ])

    # Training
    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict_quality(self, epoch_data, sfreq):
        features = extract_features(epoch_data, sfreq)
        score = self.clf.predict_proba(features)[0, 1]
        return float(score)
    
    def save(self, path):
        torch.save({
            'model_state': self.clf,
        }, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location = torch.device('cpu'))
        model = QualityModel()
        model.clf = checkpoint['model_state']
        return model
