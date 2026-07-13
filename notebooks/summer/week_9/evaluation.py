################################################################################
# Evaluates all model's performance
################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)

from models.csp_lda import CSP_LDA_Model
from models.eegnet import EEGNet_Model
from models.biot import BIOT_Model

figures = []

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

def collect_probs_and_labels(model, epochs_test, y_test):
    probs = []
    labels = []

    for i in range(len(epochs_test)):
        epoch_data = epochs_test.get_data()[i]
        prob_right = model.predict_proba(epoch_data)
        probs.append(prob_right)
        labels.append(y_test[i])

    probs = np.array(probs)
    labels = np.array(labels)
    return probs, labels

def compute_ece(probs, labels, n_bins = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    preds = (probs > 0.5).astype(int)

    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        idx = np.where((probs >= start) & (probs < end))[0]

        if len(idx) == 0:
            continue
        
        bin_conf = np.mean(probs[idx])
        bin_acc = np.mean(labels[idx] == preds[idx])
        ece += (len(idx) / len(probs)) * np.abs(bin_acc - bin_conf)

    return ece

def compute_mce(probs, labels, n_bins = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    preds = (probs > 0.5).astype(int)
    errors = []

    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        idx = np.where((probs >= start) & (probs < end))[0]

        if len(idx) == 0:
            continue
        
        bin_conf = np.mean(probs[idx])
        bin_acc = np.mean(labels[idx] == preds[idx])
        errors.append(np.abs(bin_acc - bin_conf))

    return max(errors) if errors else 0.0

def reliability_diagram(probs, labels, n_bins = 10, title = "Reliability Diagram"):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    preds = (probs > 0.5).astype(int)

    bin_confs = []
    bin_accs = []

    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        idx = np.where((probs >= start) & (probs < end))[0]

        if len(idx) == 0:
            continue

        bin_confs.append(np.mean(probs[idx]))
        bin_accs.append(np.mean(labels[idx] == preds[idx]))

    plt.figure()
    plt.plot(bin_confs, bin_accs, marker = "o", label = "Model")
    plt.plot([0, 1], [0, 1], color = "gray", label = "Perfect")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    title = title.replace(" ", "_")
    plt.savefig(f"figs/week_9/{title}.png")
    figures.append(f"{title}.png")
    plt.show(block = False)

def confidence_histograms(probs, labels, title_prefix = "Model"):
    preds = (probs > 0.5).astype(int)
    wrong = probs[labels != preds]

    plt.figure()
    plt.hist(probs, bins = 20, alpha = 0.5, label = "All Predictions")
    plt.xlabel("Confidence (P of Right Hand)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Confidence Histogram")
    plt.grid(True)
    title_prefix_underscore = title_prefix.replace(" ", "_")
    plt.savefig(f"figs/week_9/{title_prefix_underscore}_Confidence_Histogram.png")
    figures.append(f"{title_prefix_underscore}_Confidence_Histogram.png")
    plt.show(block = False)

    plt.figure()
    plt.hist(wrong, bins = 20, alpha = 0.7, color = "red", label = "Wrong Predictions")
    plt.xlabel("Confidence (P of Right Hand)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Wrong Predictions Confidence Histogram")
    plt.grid(True)
    plt.savefig(f"figs/week_9/{title_prefix_underscore}_Wrong_Predictions_Confidence_Histogram.png")
    figures.append(f"{title_prefix_underscore}_Wrong_Predictions_Confidence_Histogram.png")
    plt.show(block = False)

def safety_threshold_analysis(probs, labels, threshold = 0.75, name = "Model"):
    preds = (probs > 0.5).astype(int)
    accepted = probs >= threshold

    if np.sum(accepted) > 0:
        acc_above = np.mean(labels[accepted] == preds[accepted])
    else:
        acc_above = np.nan
    
    reject_rate = 1.0 - np.mean(accepted)

    print(f"Safety Threshold Analysis for {name}:")
    print(f"Threshold: {threshold}")
    print(f"Acceptance Rate: {np.mean(accepted):.2f}")
    print(f"Reject Rate: {reject_rate:.2f}")
    print(f"Accuracy Above Threshold: {acc_above:.2f}" if not np.isnan(acc_above) else "Accuracy Above Threshold: N/A (No accepted predictions)")

def main():
    csp_path = os.path.join(PROJECT_ROOT, "models", "csp_lda_model.pt")
    eegnet_path = os.path.join(PROJECT_ROOT, "models", "eegnet_model.pt")
    biot_path = os.path.join(PROJECT_ROOT, "models", "biot_model.pt")

    training_subjects = [1, 2, 3, 5, 6, 7, 8, 9]
    test_subject = 4

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

    print("Loading test subject...")
    epochs_test, y_test = build_epochs_for_subject(test_subject)

    # Load CSP + LDA model if available, otherwise train and save
    if os.path.exists(csp_path):
        print("Loading pre-trained CSP+LDA model...")
        csp_lda_model = CSP_LDA_Model.load(csp_path)
    else:
        print("Training CSP+LDA model...")
        csp_lda_model = CSP_LDA_Model(n_components = 4)
        csp_lda_model.fit(X_train, y_train)
        csp_lda_model.save(csp_path)
        print("CSP+LDA model saved.")
    
    # Load EEGNet model if available, otherwise train and save
    if os.path.exists(eegnet_path):
        print("Loading pre-trained EEGNet model...")
        eegnet_model = EEGNet_Model.load(eegnet_path)
    else:
        print("Training EEGNet model...")
        eegnet_model = EEGNet_Model(n_chans = n_chans, n_times = n_times)
        eegnet_model.fit(X_train, y_train, batch_size = 32, lr = 1e-3, n_epochs = 40)
        eegnet_model.save(eegnet_path)
        print("EEGNet model saved.")
    
    # Load BIOT model if available, otherwise train and save
    if os.path.exists(biot_path):
        print("Loading pre-trained BIOT model...")
        biot_model = BIOT_Model.load(biot_path)
    else:
        print("Training BIOT model...")
        biot_model = BIOT_Model(n_chans = n_chans, n_times = n_times, n_classes = 2)
        biot_model.fit(X_train, y_train, batch_size = 32, lr = 1e-3, n_epochs = 40)
        biot_model.save(biot_path)
        print("BIOT model saved.")

    results = []

    for name, model in [
        ("CSP + LDA", csp_lda_model),
        ("EEGNet", eegnet_model),
        ("BIOT", biot_model)
    ]:
        entry = {}
        entry["name"] = name

        probs, labels = collect_probs_and_labels(model, epochs_test, y_test)

        preds = (probs > 0.5).astype(int)
        entry["accuracy"] = accuracy_score(labels, preds)
        entry["mean_conf"] = np.mean(probs)
        entry["std_conf"] = np.std(probs)
        entry["ece"] = compute_ece(probs, labels)
        entry["mce"] = compute_mce(probs, labels)

        reliability_diagram(probs, labels, n_bins=10, title=f"{name} Reliability Diagram")
        confidence_histograms(probs, labels, title_prefix=name)

        threshold = 0.75
        accepted = probs >= threshold
        preds = (probs > 0.5).astype(int)

        if np.sum(accepted) > 0:
            acc_above = np.mean(labels[accepted] == preds[accepted])
        else:
            acc_above = None

        entry["threshold"] = threshold
        entry["accept_rate"] = float(np.mean(accepted))
        entry["reject_rate"] = float(1.0 - np.mean(accepted))
        entry["acc_above"] = acc_above

        results.append(entry)

    print("\n==================== Week 9 Evaluation Summary ====================\n")

    for r in results:
        print(f"=== {r['name']} ===")
        print(f"Accuracy: {r['accuracy']:.4f}")
        print(f"Mean Confidence: {r['mean_conf']:.4f}")
        print(f"Std Confidence: {r['std_conf']:.4f}")
        print(f"Expected Calibration Error (ECE): {r['ece']:.4f}")
        print(f"Maximum Calibration Error (MCE): {r['mce']:.4f}")
        print(f"Safety Threshold: {r['threshold']}")
        print(f"Acceptance Rate: {r['accept_rate']:.4f}")
        print(f"Reject Rate: {r['reject_rate']:.4f}")

        if r["acc_above"] is None:
            print("Accuracy Above Threshold: N/A (No accepted predictions)")
        else:
            print(f"Accuracy Above Threshold: {r['acc_above']:.4f}")

    figures_path = os.path.join(PROJECT_ROOT, "figs", "week_9", "generated_figs.txt")
    with open(figures_path, "w") as f:
        for fig in figures:
            f.write(f"{fig}\n")

if __name__ == "__main__":
    main()
