################################################################################
# Evaluate the performance of all models using full LOSO
################################################################################

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    classes, y_int = np.unique(y, return_inverse = True)
    n_channels = X.shape[1]

    montage = mne.channels.make_standard_montage("standard_1020")
    channel_names = montage.ch_names[:n_channels]

    info = mne.create_info(ch_names = channel_names, sfreq = 250, ch_types = "eeg")
    info.set_montage(montage)
    
    epochs = mne.EpochsArray(X, info, verbose=False)
    return epochs, y_int

def collect_probs_and_labels(model, epochs_test, y_test):
    probs, labels = [], []
    for i in range(len(epochs_test)):
        epoch_data = epochs_test.get_data()[i]
        prob_right = model.predict_proba(epoch_data)
        probs.append(prob_right)
        labels.append(y_test[i])
    return np.array(probs), np.array(labels)

def compute_ece(probs, labels, n_bins = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    preds = (probs > 0.5).astype(int)

    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        idx = np.where((probs >= start) & (probs < end))[0]
        if len(idx) == 0: continue
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
        if len(idx) == 0: continue
        bin_conf = np.mean(probs[idx])
        bin_acc = np.mean(labels[idx] == preds[idx])
        errors.append(np.abs(bin_acc - bin_conf))
    return max(errors) if errors else 0.0

def compute_split_conformal(probs, labels, alpha = 0.1, calib_frac = 0.2, random_state = 50):
    np.random.seed(random_state)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)
    n_calib = int(n_samples * calib_frac)

    calib_idx = indices[:n_calib]
    test_idx = indices[n_calib:]

    calib_probs, calib_labels = probs[calib_idx], labels[calib_idx]
    test_probs, test_labels = probs[test_idx], labels[test_idx]

    # Score = 1 - probability of the true class
    calib_true_probs = np.where(calib_labels == 1, calib_probs, 1 - calib_probs)
    calib_scores = 1 - calib_true_probs

    n = len(calib_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    q_hat = np.quantile(calib_scores, q_level, method = 'higher')

    in_set_1 = (1.0 - test_probs) <= q_hat
    in_set_0 = test_probs <= q_hat

    covered = np.where(test_labels == 1, in_set_1, in_set_0)
    empirical_coverage = np.mean(covered)
    avg_set_size = np.mean(in_set_1.astype(int) + in_set_0.astype(int))
    return empirical_coverage, avg_set_size

def plot_risk_coverage(pooled_results, eval_dir, title="Risk-Coverage Curve"):
    plt.figure(figsize=(8, 6))

    for name, data in pooled_results.items():
        probs = np.array(data["probs"])
        labels = np.array(data["labels"])

        confidences = np.maximum(probs, 1 - probs)
        preds = (probs > 0.5).astype(int)
        errors = (preds != labels).astype(int)

        sort_idx = np.argsort(confidences)[::-1]
        errors_sorted = errors[sort_idx]

        coverages = []
        risks = []
        n_samples = len(labels)

        for i in range(1, n_samples + 1):
            coverages.append(i / n_samples)
            risks.append(np.mean(errors_sorted[:i]))
        plt.plot(coverages, risks, label = name, linewidth = 2)
    plt.xlabel("Coverage (Fraction of Accepted Predictions)")
    plt.ylabel("Risk (Error Rate)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha = 0.7)
    title_safe = title.replace(" ", "_")
    file_name = f"{title_safe}.png"
    plt.savefig(os.path.join(eval_dir, file_name))
    figures.append(file_name)
    plt.close()

def symmetric_accuracy_diagram(probs, labels, eval_dir, n_bins = 10, title = "Symmetric Accuracy Diagram"):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    preds = (probs > 0.5).astype(int)
    bin_probs, bin_accs = [], []

    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        idx = np.where((probs >= start) & (probs < end))[0]
        if len(idx) == 0: continue
        bin_probs.append(np.mean(probs[idx]))
        bin_accs.append(np.mean(labels[idx] == preds[idx]))

    plt.figure()
    plt.plot(bin_probs, bin_accs, marker = "o", label = "Model")

    perfect_x = np.linspace(0.0, 1.0, 100)
    perfect_y = np.maximum(perfect_x, 1 - perfect_x)
    plt.plot(perfect_x, perfect_y, color = "gray", label = "Perfect Calibration")

    plt.xlabel("Predicted Probability P(Right Hand) [Left < 0.5]")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    title_safe = title.replace(" ", "_")
    file_name = f"{title_safe}.png"
    plt.savefig(os.path.join(eval_dir, file_name))
    figures.append(file_name)
    plt.close()

def confidence_calibration_diagram(probs, labels, eval_dir, n_bins = 10, title = "Confidence Calibration Diagram"):
    confidences = np.maximum(probs, 1 - probs)
    preds = (probs > 0.5).astype(int)

    bins = np.linspace(0.5, 1.0, n_bins + 1)
    bin_confs, bin_accs = [], []

    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        idx = np.where((confidences >= start) & (confidences < end))[0]
        if len(idx) == 0: continue
        bin_confs.append(np.mean(confidences[idx]))
        bin_accs.append(np.mean(labels[idx] == preds[idx]))
    
    plt.figure()
    plt.plot(bin_confs, bin_accs, marker = "o", label = "Model")

    plt.plot([0.5, 1.0], [0.5, 1.0], color = "gray", label = "Perfect Calibration")

    plt.xlabel("Confidence Max")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    title_safe = title.replace(" ", "_")
    file_name = f"{title_safe}.png"
    plt.savefig(os.path.join(eval_dir, file_name))
    figures.append(file_name)
    plt.close()

def confidence_histograms(probs, labels, eval_dir, title_prefix = "Model"):
    preds = (probs > 0.5).astype(int)
    wrong = probs[labels != preds]
    title_prefix_safe = title_prefix.replace(" ", "_")

    plt.figure()
    plt.hist(probs, bins = 20, alpha = 0.5, label = "All Predictions")
    plt.xlabel("Confidence (P of Right Hand)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Confidence Histogram")
    plt.grid(True)
    file_name_1 = f"{title_prefix_safe}_Confidence_Histogram.png"
    plt.savefig(os.path.join(eval_dir, file_name_1))
    figures.append(file_name_1)
    plt.close()

    plt.figure()
    plt.hist(wrong, bins = 20, alpha = 0.7, color = "red", label = "Wrong Predictions")
    plt.xlabel("Confidence (P of Right Hand)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} Wrong Predictions Confidence Histogram")
    plt.grid(True)
    file_name_2 = f"{title_prefix_safe}_Wrong_Predictions_Confidence_Histogram.png"
    plt.savefig(os.path.join(eval_dir, file_name_2))
    figures.append(file_name_2)
    plt.close()

def main():
    eval_dir = os.path.join(PROJECT_ROOT, "figs", "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    subjects = list(range(1, 10))
    model_names = ["CSP + LDA", "EEGNet", "BIOT"]
    
    # Store per-subject metrics for averaging later
    subject_metrics = {name: [] for name in model_names}
    
    # Store pooled outputs across all subjects for plot generation
    pooled_results = {name: {"probs": [], "labels": []} for name in model_names}

    for test_subj in subjects:
        print(f"\nRunning Test Subject {test_subj}")
        training_subjects = [s for s in subjects if s != test_subj]

        train_epochs_list, train_labels_list = [], []
        for subj in training_subjects:
            epochs, labels = build_epochs_for_subject(subj)
            train_epochs_list.append(epochs)
            train_labels_list.append(labels)

        epochs_train = mne.concatenate_epochs(train_epochs_list)
        y_train = np.concatenate(train_labels_list)
        X_train = epochs_train.get_data()
        
        n_chans = X_train.shape[1]
        n_times = X_train.shape[2]

        epochs_test, y_test = build_epochs_for_subject(test_subj)

        print("Training CSP+LDA model...")
        csp_lda_model = CSP_LDA_Model(n_components = 4)
        csp_lda_model.fit(X_train, y_train)
        
        print("Training EEGNet model...")
        eegnet_model = EEGNet_Model(n_chans = n_chans, n_times = n_times)
        eegnet_model.fit(X_train, y_train)
        
        print("Training BIOT model...")
        biot_model = BIOT_Model(n_chans = n_chans, n_times = n_times, n_classes = 2)
        biot_model.fit(X_train, y_train)

        for name, model in [("CSP + LDA", csp_lda_model), ("EEGNet", eegnet_model), ("BIOT", biot_model)]:
            probs, labels = collect_probs_and_labels(model, epochs_test, y_test)
            
            # Save raw data for pooled figures
            pooled_results[name]["probs"].extend(probs)
            pooled_results[name]["labels"].extend(labels)

            # Compute subject-level evaluation metrics
            preds = (probs > 0.5).astype(int)
            acc = accuracy_score(labels, preds)
            m_conf = np.mean(probs)
            s_conf = np.std(probs)
            ece = compute_ece(probs, labels)
            mce = compute_mce(probs, labels)

            threshold = 0.75
            accepted = probs >= threshold
            accept_rate = float(np.mean(accepted))
            reject_rate = float(1.0 - accept_rate)
            acc_above = float(np.mean(labels[accepted] == preds[accepted])) if np.sum(accepted) > 0 else np.nan

            subject_metrics[name].append({
                "accuracy": acc,
                "mean_conf": m_conf,
                "std_conf": s_conf,
                "ece": ece,
                "mce": mce,
                "accept_rate": accept_rate,
                "reject_rate": reject_rate,
                "acc_above": acc_above
            })

    # Generate pooled diagrams using full-dataset distribution
    for name in model_names:
        all_probs = np.array(pooled_results[name]["probs"])
        all_labels = np.array(pooled_results[name]["labels"])
        symmetric_accuracy_diagram(all_probs, all_labels, eval_dir, n_bins=10, title=f"{name} Symmetric Accuracy Diagram")
        confidence_calibration_diagram(all_probs, all_labels, eval_dir, n_bins=10, title=f"{name} Confidence Calibration Diagram")
        confidence_histograms(all_probs, all_labels, eval_dir, title_prefix=name)

    plot_risk_coverage(pooled_results, eval_dir, title="Risk Coverage Curve")

    print("\n==================== Evaluation Summary ====================\n")

    conformal_alpha = 0.1

    for name in model_names:
        metrics_list = subject_metrics[name]
        
        avg_acc = np.mean([m["accuracy"] for m in metrics_list])
        avg_m_conf = np.mean([m["mean_conf"] for m in metrics_list])
        avg_s_conf = np.mean([m["std_conf"] for m in metrics_list])
        avg_ece = np.mean([m["ece"] for m in metrics_list])
        avg_mce = np.mean([m["mce"] for m in metrics_list])
        avg_accept = np.mean([m["accept_rate"] for m in metrics_list])
        avg_reject = np.mean([m["reject_rate"] for m in metrics_list])
        
        acc_above_vals = [m["acc_above"] for m in metrics_list if not np.isnan(m["acc_above"])]
        avg_acc_above = np.mean(acc_above_vals) if len(acc_above_vals) > 0 else None
        emp_cov, avg_size = compute_split_conformal(all_probs, all_labels, alpha=conformal_alpha)

        print(f"=== {name} ===")
        print(f"Accuracy: {avg_acc:.4f}")
        print(f"Mean Confidence: {avg_m_conf:.4f}")
        print(f"Std Confidence: {avg_s_conf:.4f}")
        print(f"Expected Calibration Error (ECE): {avg_ece:.4f}")
        print(f"Maximum Calibration Error (MCE): {avg_mce:.4f}")
        print(f"Safety Threshold: 0.75")
        print(f"Acceptance Rate: {avg_accept:.4f}")
        print(f"Reject Rate: {avg_reject:.4f}")
        print(f"Empirical Coverage: {emp_cov:.4f}")
        print(f"Average Set Size: {avg_size:.4f}")

        if avg_acc_above is None:
            print("Accuracy Above Threshold: N/A (No accepted predictions across subjects)")
        else:
            print(f"Accuracy Above Threshold: {avg_acc_above:.4f}")
        print()

    figures_path = os.path.join(eval_dir, "generated_figs.txt")
    with open(figures_path, "w") as f:
        for fig in figures:
            f.write(f"{fig}\n")

if __name__ == "__main__":
    main()