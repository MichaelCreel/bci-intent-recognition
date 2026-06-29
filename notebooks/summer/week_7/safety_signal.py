################################################################################
# Determines whether a command should be executed based on the previous k epochs
################################################################################

import numpy as np
from collections import deque
import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# Pre-load the model for intent classification
def load_intent_model():
    class IntentModel:
        def predict_proba(self, epoch_data):
            # epoch_data is shape (n_channels, n_times)
            return 0.8 # placeholder confidence score
    return IntentModel()

# Pre-load the model for quality classification
def load_quality_model():
    class QualityModel:
        def compute_quality_score(self, epoch_data, sfreq):
            return 0.9 # placeholder quality score
    return QualityModel()

# Compute temporal stability between 0 and 1
def compute_temporal_stability(conf_history, min_len = 3):
    if len(conf_history) < min_len:
        return 0.0 # Consider low stability with little data
    
    arr = np.array(conf_history)
    var = np.var(arr)

    # Map variance to stability
    # High variance = low stability
    # Low variance = high stability
    max_var = 0.25 # Threshold for stable variance
    stability = 1.0 - np.clip(var / max_var, 0.0, 1.0)
    
    return float(stability)

# Calculate the overall score using different modes and weights
# min mode: The system relies on the weakest signal
#   if one score is low, the overall score is low
# mean mode: The system averages the scores
#   all scores contribute equally to the overall score
# wmean mode: The system averages weighted scores
#   all scores contribute, but more important scores have a larger impact on the overall score
#   wmean requires additional inputs as weights or it will function the same as mean mode
# geo mode: The system uses the geometric mean of the scores
#   all scores contribute equally, but low scores have a large effect on the overall score
# wgeo mode: The system uses the weighted geometric mean of the scores
#   all scores contribute, but more important scores have a larger impact on the overall score
def compute_overall_score(intent_confidence, quality_score, stability, mode = "min", weights = (1.0, 1.0, 1.0)):
    C = float(intent_confidence)
    Q = float(quality_score)
    S = float(stability)

    # min mode
    if mode == "min":
        safety = min(C, Q, S)
    # mean mode
    elif mode == "mean":
        safety = (C + Q + S) / 3.0
    # wmean mode
    elif mode == "wmean":
        wC, wQ, wS = weights
        safety = (wC * C + wQ * Q + wS * S) / (wC + wQ + wS)
    # geo mode
    elif mode == "geo":
        C_ = np.clip(C, 1e-6, 1.0)
        Q_ = np.clip(Q, 1e-6, 1.0)
        S_ = np.clip(S, 1e-6, 1.0)
        safety = (C_ * Q_ * S_) ** (1.0 / 3.0)
    # wgeo mode
    elif mode == "wgeo":
        wC, wQ, wS = weights
        C_ = np.clip(C, 1e-6, 1.0)
        Q_ = np.clip(Q, 1e-6, 1.0)
        S_ = np.clip(S, 1e-6, 1.0)
        safety = (C_**wC * Q_**wQ * S_**wS) ** (1.0 / (wC + wQ + wS))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    return float(np.clip(safety, 0.0, 1.0))

# Determine if a command should be executed based on the safety score
# This can be replaced with a damping function based on the safety score
def should_execute_command(safety_score, threshold = 0.75):
    return safety_score >= threshold

# EXAMPLE USAGE

def build_epochs_for_subject(subject_id):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes = 2,
        events = ['left_hand', 'right_hand'],
        fmin = 8,
        fmax = 30
    )

    X, y, meta = paradigm.get_data(dataset = dataset, subjects = [subject_id])
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
    return epochs

if __name__ == "__main__":
    training_subjects = [1, 3, 7]
    testing_subject = 4

    print("Loading training subjects...")
    train_epochs_list = []

    for subj in training_subjects:
        epochs = build_epochs_for_subject(subj)
        train_epochs_list.append(epochs)
    
    epochs_train = mne.concatenate_epochs(train_epochs_list)
    sfreq = epochs_train.info['sfreq']

    # Train intent classifier
    print("Training intent classifier...")
    intent_model = load_intent_model()
    # Call training here

    # Train quality classifier
    print("Training quality classifier...")
    quality_model = load_quality_model()
    # Call training here

    # Load testing subject
    print("Loading testing subject...")
    epochs_test = build_epochs_for_subject(testing_subject)
    sfreq = epochs_test.info['sfreq']

    intent_conf_history = deque(maxlen = 5)

    print("Simulating safety score calculation...")
    for i in range(len(epochs_test)):
        epoch_data = epochs_test.get_data()[i] 

        intent_confidence = intent_model.predict_proba(epoch_data)
        quality_score = quality_model.compute_quality_score(epoch_data, sfreq)

        intent_conf_history.append(intent_confidence)
        stability = compute_temporal_stability(intent_conf_history)

        safety_score = compute_overall_score(
            intent_confidence,
            quality_score,
            stability,
            mode = "min"
        )

        execute = should_execute_command(safety_score, threshold = 0.75)

        print(
            f"Epoch {i+1}/{len(epochs_test)}: "
            f"Intent Confidence = {intent_confidence:.3f}, "
            f"Quality Score = {quality_score:.3f}, "
            f"Stability = {stability:.3f}, "
            f"Safety Score = {safety_score:.3f}, "
            f"Execute Command = {'Yes' if execute else 'No'}"
        )
