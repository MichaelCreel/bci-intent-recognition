################################################################################
# Tests all models to compare their performance on sample datasets
################################################################################

import os
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score
import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from models.csp_lda import CSP_LDA_Model
from models.eegnet import EEGNet_Model
from models.biot import BIOT_Model

def build_epochs_for_subject(subject_id):
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes = 2,
        events = ["left_hand", "right_hand"],
        fmin = 8,
        fmax = 30,
    )

    X, y, meta = paradigm.get_data(dataset = dataset, subjects = [subject_id])
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

def evaluate_model(model, epochs_test, y_test):
    preds = []
    confs = []

    for i in range(len(epochs_test)):
        epoch_data = epochs_test.get_data()[i]
        prob_right = model.predict_proba(epoch_data)
        confs.append(prob_right)
        preds.append(1 if prob_right >= 0.5 else 0)

    preds = np.array(preds)
    y_test = np.array(y_test)

    accuracy = accuracy_score(y_test, preds)
    mean_conf = np.mean(confs)
    std_conf = np.std(confs)

    return accuracy, mean_conf, std_conf

def main():
    N_RUNS = int(input("How many times should each model be tested? "))

    training_subjects = [1, 2, 3, 5, 6, 7, 8, 9]
    test_subject = 4

    # Load training subjects
    print("Loading training subjects...")
    train_epochs_list = []
    train_labels_list = []

    for subj in training_subjects:
        epochs, labels = build_epochs_for_subject(subj)
        train_epochs_list.append(epochs)
        train_labels_list.append(labels)

    epochs_train = mne.concatenate_epochs(train_epochs_list)
    y_train = np.concatenate(train_labels_list)

    X_train = epochs_train.get_data()
    n_chans = X_train.shape[1]
    n_times = X_train.shape[2]

    # Load test subject
    print("Loading test subject...")
    epochs_test, y_test = build_epochs_for_subject(test_subject)

    # Results Arrays
    results_csp = []
    results_eegnet = []
    results_biot = []

    for run in range(N_RUNS):
        print(f"Run {run + 1}/{N_RUNS}")

        # Train and evaluate CSP + LDA
        print("Training CSP + LDA...")
        csp_model = CSP_LDA_Model()
        csp_model.fit(X_train, y_train)
        acc_csp, mean_conf_csp, std_conf_csp = evaluate_model(csp_model, epochs_test, y_test)
        results_csp.append((acc_csp, mean_conf_csp, std_conf_csp))
        print(f"CSP + LDA Accuracy: {acc_csp:.4f}, Mean Confidence: {mean_conf_csp:.4f}, Std Confidence: {std_conf_csp:.4f}")

        # Train and evaluate EEGNet
        print("Training EEGNet...")
        eegnet_model = EEGNet_Model(n_chans = n_chans, n_times = n_times)
        eegnet_model.fit(X_train, y_train, batch_size = 32, lr = 1e-3, n_epochs = 40)
        acc_eegnet, mean_conf_eegnet, std_conf_eegnet = evaluate_model(eegnet_model, epochs_test, y_test)
        results_eegnet.append((acc_eegnet, mean_conf_eegnet, std_conf_eegnet))
        print(f"EEGNet Accuracy: {acc_eegnet:.4f}, Mean Confidence: {mean_conf_eegnet:.4f}, Std Confidence: {std_conf_eegnet:.4f}")

        # Train and evaluate BIOT
        print("Training BIOT...")
        biot_model = BIOT_Model(n_chans = n_chans, n_times = n_times, n_classes = 2)
        biot_model.fit(X_train, y_train, batch_size = 32, lr = 1e-3, n_epochs = 40)
        acc_biot, mean_conf_biot, std_conf_biot = evaluate_model(biot_model, epochs_test, y_test)
        results_biot.append((acc_biot, mean_conf_biot, std_conf_biot))
        print(f"BIOT Accuracy: {acc_biot:.4f}, Mean Confidence: {mean_conf_biot:.4f}, Std Confidence: {std_conf_biot:.4f}")

    def summarize(name, arr):
        arr = np.array(arr)
        print(f"\n=== {name} Summary Across {N_RUNS} Runs ===")
        print(f"Mean Accuracy: {np.mean(arr[:, 0]):.4f} std: {np.std(arr[:, 0]):.4f}")
        print(f"Mean Confidence: {np.mean(arr[:, 1]):.4f} std: {np.std(arr[:, 1]):.4f}")
        print(f"Std Confidence: {np.mean(arr[:, 2]):.4f} std: {np.std(arr[:, 2]):.4f}")

    summarize("CSP + LDA", results_csp)
    summarize("EEGNet", results_eegnet)
    summarize("BIOT", results_biot)

if __name__ == "__main__":
    main()
