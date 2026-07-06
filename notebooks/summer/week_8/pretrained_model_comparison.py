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
