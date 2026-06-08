################################################################################
# Trains one EEGNet model on multiple subjects and evaluates performance
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

        X_test = X_all[subject]
        y_test = y_all[subject]

        print(f"Training size: {X_train.shape}, Test size: {X_test.shape}")

        # Create DataLoaders
        train_data = data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        test_data = data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

        train_loader = data.DataLoader(train_data, batch_size = batch_size, shuffle = True)
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

        # Evaluate model
        model.eval()
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                predicted = preds.argmax(dim = 1)

                preds_all.extend(predicted.cpu().numpy())
                labels_all.extend(yb.cpu().numpy())

        preds_all = np.array(preds_all)
        labels_all = np.array(labels_all)

        accuracy = (preds_all == labels_all).mean()
        cm = confusion_matrix(labels_all, preds_all)

        print(f"Subject {subject}: Accuracy = {accuracy:.2f}, Confusion Matrix:\n{cm}")
        print(f"Confusion Matrix:\n{cm}")

        accuracys.append(accuracy)
        confusion_matrices.append((subject, cm))

    return accuracys, confusion_matrices

generated_figs = []

if __name__ == "__main__":
    subjects = [1, 2, 3]
    
    accuracys, cms = train_eegnet(subjects = subjects)

    print("Accuracies:", accuracys)
    print("Average Accuracy:", np.mean(accuracys))

    for subject, cm in cms:
        plt.imshow(cm, cmap = "Blues")
        plt.title(f"Multi-Subject Confusion Matrix - Subject {subject}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.show(block = False)
        plt.savefig(f"figs/week_4/cross_subject_eegnet_cm_subject_{subject}.png")
        generated_figs.append(f"cross_subject_eegnet_cm_subject_{subject}.png")
        with open("figs/week_4/cross_subject_generated_figs.txt", "w") as f:
            f.write("\n".join(generated_figs))