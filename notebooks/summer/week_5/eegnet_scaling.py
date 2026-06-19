################################################################################
# Applies softmax confidence and temperature scaling to multi-subject EEGNet
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.utils.data as data
from braindecode.models import EEGNet
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

# Class for temperature scaling
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature
    
def fit_temperature(model, val_loader, device):
    model.eval()
    scaler = TemperatureScaler().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr = 0.01, max_iter = 50)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            logits_list.append(logits)
            labels_list.append(yb)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    def eval():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    return scaler

# Compute Expected Calibration Error (ECE)
def compute_ece(probs, labels, n_bins = 5):
    confidences = np.max(probs, axis = 1)
    predictions = np.argmax(probs, axis = 1)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            average_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(average_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

# Create reliability diagram
# Increasing the number of bins will make the diagram more jagged due to data scarcity
# Increasing the number of bins will also increase the number of empty bins
def reliability_diagram(probs, labels, n_bins = 5, title = "Reliability Diagram (Improper Title)"):
    confidences = np.max(probs, axis = 1)
    predictions = np.argmax(probs, axis = 1)

    bins = np.linspace(0, 1, n_bins + 1)
    accuracies = []
    confs = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(in_bin):
            accuracies.append(np.mean(predictions[in_bin] == labels[in_bin]))
            confs.append(np.mean(confidences[in_bin]))
        # Accounts for empty bins
        # This can make relability diagrams look odd with jumps to the origin
        # else:
        #     accuracies.append(0)
        #     confs.append(0)

    plt.figure()
    plt.plot(confs, accuracies, marker = 'o')
    plt.plot([0, 1], [0, 1], linestyle = '--')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.show(block = False)
    plt.savefig(f"figs/week_5/{title.replace(' ', '_').lower()}.png")

# Load multiple subjects data
# Filters data between 8 and 30 Hz
def load_subjects(subjects):
    dataset = PhysionetMI()
    paradigm = MotorImagery(
        n_classes = 2,
        events = ['left_hand', 'right_hand'],
        fmin = 8,
        fmax = 30
    )

    X_all = {}
    y_all = {}

    for subject in subjects:
        print(f"Loading subject {subject}...")
        X, y, meta = paradigm.get_data(dataset = dataset, subjects = [subject])

        classes, y_int = np.unique(y, return_inverse = True)

        X_all[subject] = X.astype(np.float32)
        y_all[subject] = y_int.astype(np.int64)

        print(f"Subject {subject}\n- X shape: {X.shape}\n- y counts: {np.bincount(y_int)}")
    
    return X_all, y_all

# Trains EEGNet on multiple subjects and evaluates performance
# Returns the accuracy of the model and the confusion matrix
def train_eegnet(subjects, n_epochs = 40, batch_size = 32, lr = 1e-3):
    print(f"Training EEGNet on subjects {subjects}...")

    X_all, y_all = load_subjects(subjects)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    accuracys = []
    confusion_matrices = []

    for subject in subjects:
        X_train = []
        y_train = []

        # Build training data set from all subjects excluding current
        for subj in subjects:
            if subj != subject:
                X_train.append(X_all[subj])
                y_train.append(y_all[subj])
    
        X_train = np.concatenate(X_train, axis = 0)
        y_train = np.concatenate(y_train, axis = 0)

        # Split training into train/val for calibrating
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size = 0.2, stratify = y_train
        )

        X_test = X_all[subject]
        y_test = y_all[subject]

        print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")

        # Create DataLoaders
        train_data = data.TensorDataset(
            torch.tensor(X_train_split, dtype=torch.float32),
            torch.tensor(y_train_split, dtype=torch.long)
        )
        val_data = data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        test_data = data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
        val_loader = data.DataLoader(val_data, batch_size = batch_size)
        test_loader = data.DataLoader(test_data, batch_size = batch_size)

        # Create EEGNet model
        model = EEGNet(
            n_chans = X_train.shape[1],
            n_outputs = 2,
            n_times = X_train.shape[2]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        criterion = nn.CrossEntropyLoss()

        # Train model
        for epoch in range(n_epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        # Apply temperature scaling
        scaler = fit_temperature(model, val_loader, device)

        # Evaluate model
        model.eval()
        preds_all = []
        labels_all = []
        probs_all = []
        probs_pre_scale_all = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)

                logits = model(xb)

                # Fill pre-scaling arrays
                probs_before = torch.softmax(logits, dim = 1)
                probs_pre_scale_all.extend(probs_before.cpu().numpy())

                # Apply temperature scaling and get probabilities
                scaled_logits = scaler(logits)
                probs = torch.softmax(scaled_logits, dim = 1)

                predicted = probs.argmax(dim = 1)

                preds_all.extend(predicted.cpu().numpy())
                labels_all.extend(yb.cpu().numpy())
                probs_all.extend(probs.cpu().numpy())

        ece_pre_scale = compute_ece(np.array(probs_pre_scale_all), np.array(labels_all))
        ece_after_scale = compute_ece(np.array(probs_all), np.array(labels_all))

        reliability_diagram(
            np.array(probs_pre_scale_all), np.array(labels_all),
            title = f"Subject {subject} Reliability - Before Scaling"
        )
        reliability_diagram(
            np.array(probs_all), np.array(labels_all),
            title = f"Subject {subject} Reliability - After Scaling"
        )

        print(f"Subject {subject}:\n - ECE before scaling: {ece_pre_scale:.4f}\n - ECE after scaling: {ece_after_scale:.4f}")
        
        # append the file names of generated reliability diagrams to the generated_figs.txt file
        with open("./figs/week_5/generated_figs.txt", "a") as f:
            f.write(f"figs/week_5/subject_{subject}_reliability_-_before_scaling.png\n")
            f.write(f"figs/week_5/subject_{subject}_reliability_-_after_scaling.png\n")

if __name__ == "__main__":
    with open("./figs/week_5/generated_figs.txt", "w") as f:
        f.write("")

    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_eegnet(subjects = subjects)