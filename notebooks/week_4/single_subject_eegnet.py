################################################################################
# Trains Multiple EEGNet models each using a single subject
################################################################################

import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from braindecode.models import EEGNet

# Load related runs from a subject
# Runs 3 and 7 are left hand, runs 4 and 8 are right hand
# Runs 7 and 8 are imagined movements
# Filters data between 8 and 30 Hz
def load_subject(subject, runs = [3, 4, 7, 8]):
    print(f"Loading runs {runs} for subject {subject}...")
    # Load raw data for the subject and concatenate runs
    files = eegbci.load_data(subject, runs=runs)
    raws = [read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)

    # Filter the data between 8 and 30 Hz
    filtered = raw.copy().filter(8, 30)

    # Extract events
    events, event_id = mne.events_from_annotations(filtered)

    # Map 'T1' to 'left_hand' and 'T2' to 'right_hand'
    mapping = {'T1': 'left_hand', 'T2': 'right_hand'}
    filtered.set_annotations(filtered.annotations.rename(mapping))
    events, event_id = mne.events_from_annotations(filtered)

    # Create epochs
    epochs = mne.Epochs(
        filtered,
        events,
        event_id = event_id,
        tmin = 0,
        tmax = 4,
        baseline = None,
        preload = True
    )

    # Extract left and right hand epochs
    epochs_left = epochs['left_hand']
    epochs_right = epochs['right_hand']

    # Combine left and right hand epochs
    epochs_lr = mne.concatenate_epochs([epochs_left, epochs_right])
    labels = np.concatenate([
        np.zeros(len(epochs_left)), # Left hand = 0
        np.ones(len(epochs_right))  # Right hand = 1
    ])

    X = epochs_lr.get_data()
    y = labels.astype(int)

    return X, y

# Trains EEGNet on a single subject and evaluates performance
# Returns the accuracy of the model and the confusion matrix
def train_eegnet(subject):
    print(f"Training EEGNet on subject {subject}...")

    X, y = load_subject(subject)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, shuffle = True, stratify = y
    )

    # Convert to PyTorch tensors
    train_data = data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_data = data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = data.Dataloader(train_data, batch_size = 32, shuffle = True)
    test_loader = data.DataLoader(test_data, batch_size = 32)

    # Create EEGNet model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGNet(
        n_chans = X.shape[1],
        n_outputs = 2,
        n_times = X.shape[2]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train model
    epochs = 40
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predicted = preds.argmax(dim = 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

            preds_all.extend(predicted.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())
    
    accuracy = correct / total
    print(f"Subject {subject} - Test Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(labels_all, preds_all)
    print("Confusion Matrix:")
    print(cm)

    plt.imshow(cm, cmap = "Blues")
    plt.title(f"Subject {subject} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.show()

    return accuracy, cm
