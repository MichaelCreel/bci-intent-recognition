################################################################################
# Implemenets competition using CSP and LDA
################################################################################

import mne
from mne import data
from mne.datasets import eegbci
from mne.io import read_raw_edf
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Initialize data for subject 1
runs = [3, 4, 7, 8] # Left hand: runs 3 and 7, Right hand: runs 4 and 8; 7 and 8 are imagined movements
files = eegbci.load_data(1, runs=runs) # Subject 1
raws = [read_raw_edf(f, preload=True) for f in files]
raw = mne.concatenate_raws(raws)

# Filter Data between 8 and 30 Hz
raw_filtered = raw.copy().filter(8, 30)
print(raw_filtered)

# Extract events
events, event_id = mne.events_from_annotations(raw_filtered)
print("Event IDs:", event_id)
print(events[:10])

# Create epochs
epochs = mne.Epochs(
    raw_filtered,
    events,
    event_id = event_id,
    tmin = 0,
    tmax = 4,
    baseline = None,
    preload = True
)

# Separate epochs for left and right hand
epochs_left = epochs['T1']
epochs_right = epochs['T2']

# Combine epochs for left and right hands and create labels
epochs_lr = mne.concatenate_epochs([epochs_left, epochs_right])
labels = np.concatenate([
    np.zeros(len(epochs_left)), # Left hand = 0
    np.ones(len(epochs_right))  # Right hand = 1
])

# Prepare data for CSP
X = epochs_lr.get_data()

# Create CSP and LDA pipeline
clf = Pipeline([
    ('CSP', CSP(n_components=4, reg=None, log=True)),
    ('LDA', LinearDiscriminantAnalysis())
])

# Evaluate accuracy on training data
scores = cross_val_score(clf, X, labels, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Accuracy: {scores.mean():.2f}")
print(f"Left-hand epochs: {len(epochs_left)}")
print(f"Right-hand epochs: {len(epochs_right)}")