################################################################################
# Applies softmax confidence and temperature scaling to CSP + LDA pipeline
################################################################################

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Class for temperature scaling
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature
    
def fit_temperature(logits, labels):
    logits = torch.tensor(logits, dtype = torch.float32)
    labels = torch.tensor(labels, dtype = torch.long)

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr = 0.01, max_iter = 50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

# Compute Expected Calibration Error (ECE)
def compute_ece(probs, labels, n_bins = 5):
    confidences = np.max(probs, axis = 1)
    prediction = np.argmax(probs, axis = 1)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(prediction[in_bin] == labels[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

# Create Reliability Diagram
# Increasing the number of bins will make the diagram more jagged due to data scarcity
# Increasing the number of bins will also increase the number of empty bins
def reliability_diagram(probs, labels, n_bins = 5, title = "Reliability Diagram (Improper Title)"):
    confidences = np.max(probs, axis = 1)
    predictions = np.argmax(probs, axis = 1)

    bins = np.linspace(0, 1, n_bins + 1)
    accuracies = []
    confidences_in_bins = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(in_bin):
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            confidences_in_bins.append(confidence_in_bin)
            accuracies.append(accuracy_in_bin)
        # Accounts for empty bins
        # This can make relability diagrams look odd with jumps to the origin
        # else:
        #     accuracy_in_bin = 0
        #     confidence_in_bin = 0

    plt.figure()
    plt.plot(confidences_in_bins, accuracies, marker = 'o')
    plt.plot([0, 1], [0, 1], linestyle = '--')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.show(block = False)
    plt.savefig(f"figs/week_5/csp_lda_{title.replace(' ', '_').lower()}.png")

# CSP + LDA Pipeline
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]

dataset = BNCI2014_001()
paradigm = MotorImagery(
    n_classes = 2,
    events = ['left_hand', 'right_hand'],
    fmin = 8,
    fmax = 30
)

scores_all = []

for subject in subjects:
    print(f"Building pipeline for subject {subject}...")
    
    X_all = {}
    y_all = {}

    for subj in subjects:
        X, y, meta = paradigm.get_data(dataset = dataset, subjects = [subj])
        _, y_int = np.unique(y, return_inverse = True)
        X_all[subj] = X.astype(np.float32)
        y_all[subj] = y_int.astype(np.int64)

    # Build Train Set
    X_train = []
    y_train = []

    for subj in subjects:
        if subj != subject:
            X_train.append(X_all[subj])
            y_train.append(y_all[subj])
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    # Split training into train/val for calibrating
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size = 0.2, stratify = y_train
    )

    # Build Test Set
    X_test = X_all[subject]
    y_test = y_all[subject]

    # Build CSP + LDA Pipeline
    csp = CSP(n_components = 6, log = True)
    lda = LinearDiscriminantAnalysis()

    X_train_csp = csp.fit_transform(X_train_split, y_train_split)
    X_val_csp = csp.transform(X_val)
    X_test_csp = csp.transform(X_test)

    lda.fit(X_train_csp, y_train_split)

    # Apply Temperature Scaling
    logits_val = lda.decision_function(X_val_csp)
    logits_val = np.column_stack([-logits_val, logits_val])
    scaler = fit_temperature(logits_val, y_val)

    # Test logits
    logits_test = lda.decision_function(X_test_csp)
    logits_test = np.column_stack([-logits_test, logits_test])

    # Probabilities before scaling
    prob_pre_scale_all = F.softmax(torch.tensor(logits_test), dim = 1).numpy()

    # Probabilities after scaling
    scaled_logits = scaler(torch.tensor(logits_test, dtype = torch.float32)).detach().numpy()
    probs_scaled_all = F.softmax(torch.tensor(scaled_logits), dim = 1).numpy()

    # Compute ECE before and after scaling
    ece_pre_scale = compute_ece(prob_pre_scale_all, y_test)
    ece_after_scale = compute_ece(probs_scaled_all, y_test)

    print(f"Subject {subject}:\n - ECE before scaling: {ece_pre_scale:.4f}\n - ECE after scaling: {ece_after_scale:.4f}")

    # Create reliability diagrams before and after scaling
    reliability_diagram(prob_pre_scale_all, y_test, title = f"Subject {subject} Reliability - Before Scaling")
    reliability_diagram(probs_scaled_all, y_test, title = f"Subject {subject} Reliability - After Scaling")

    with open("./figs/week_5/csp_lda_generated_figs.txt", "a") as f:
        f.write(f"csp_lda_subject_{subject}_reliability_-_before_scaling.png\n")
        f.write(f"csp_lda_subject_{subject}_reliability_-_after_scaling.png\n")
