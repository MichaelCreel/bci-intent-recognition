################################################################################
# Determines the quality of an epoch
################################################################################

import numpy as np
import mne
from autoreject import AutoReject
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Extract the features of the epoch
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

    # High frequency
    fft = np.abs(np.fft.rfft(epoch_data, axis = 1))
    freqs = np.fft.rfftfreq(epoch_data.shape[1], 1/sfreq)

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

# Build dataset with autoreject labels
def build_dataset(epochs):
    sfreq = epochs.info['sfreq']

    ar = AutoReject()

    # Decrease consensus value to make AutoReject more scrutinous in bad epoch labeling
    ar = AutoReject(consensus = [0.5], n_interpolate = [1, 2, 3])

    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs) # Good = 1, Bad = 0

    labels = (~reject_log.bad_epochs).astype(int)

    X = []
    for i in range(len(epochs)):
        epoch_data = epochs.get_data()[i]
        features = extract_features(epoch_data, sfreq)
        X.append(features)

    X = np.vstack(X)
    y = labels

    return X, y

# Train a classifier to determine epoch quality
def train_classifier(X, y):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators = 200,
            max_depth = 6,
            class_weight = "balanced",
            random_state = 42
        ))
    ])

    clf.fit(X, y)
    return clf

# Compute quality score for an epoch
def compute_quality_score(epoch_data, sfreq, clf):
    features = extract_features(epoch_data, sfreq).reshape(1, -1)
    score = clf.predict_proba(features)[0][1]
    return float(score)

# Testing using PhysioNet data
if __name__ == "__main__":
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes = 2,
        events = ['left_hand', 'right_hand'],
        fmin = 8,
        fmax = 30
    )

    training_subjects = [1, 2, 3]
    testing_subjects = [8, 9]

    # Load training subjects
    X_train, y_train, meta_train = paradigm.get_data(
        dataset = dataset,
        subjects = training_subjects
    )

    n_channels = X_train.shape[1]

    # Build montage
    montage = mne.channels.make_standard_montage("standard_1020")
    channel_names = montage.ch_names[:n_channels]

    # Create info
    info_train = mne.create_info(
        ch_names = channel_names,
        sfreq = 250,
        ch_types = "eeg"
    )
    info_train.set_montage(montage)

    epochs_train = mne.EpochsArray(X_train, info_train)

    # Build dataset
    X_features, y_labels = build_dataset(epochs_train)
    print("Unique labels:", np.unique(y_labels, return_counts = True))

    # Train classifier
    clf = train_classifier(X_features, y_labels)

    # Load testing subjects
    X_test, y_test, meta_test = paradigm.get_data(
        dataset = dataset,
        subjects = testing_subjects
    )

    info_test = mne.create_info(
        ch_names = channel_names,
        sfreq = 250,
        ch_types = "eeg"
    )
    info_test.set_montage(montage)

    epochs_test = mne.EpochsArray(X_test, info_test)

    # Compute quality score
    test_epoch = epochs_test.get_data()[0]
    score = compute_quality_score(test_epoch, epochs_test.info["sfreq"], clf)

    print(f"Epoch quality score: {score:.4f}")
