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
