################################################################################
# Trains Multiple EEGNet models each using a single subject
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

# Load related runs from a subject
# Runs 3 and 7 are left hand, runs 4 and 8 are right hand
# Runs 7 and 8 are imagined movements
# Filters data between 8 and 30 Hz
def load_subject(subject):
    print(f"Loading subject {subject}...")

    dataset = PhysionetMI()
    paradigm = MotorImagery(
        n_classes = 2,
        events = ['left_hand', 'right_hand'],
        fmin = 8,
        fmax = 30
    )

    X, y, meta = paradigm.get_data(dataset = dataset, subjects = [subject])

    classes, y_int = np.unique(y, return_inverse = True)

    print(f"Classes: {classes}")
    print(f"Label counts: {np.bincount(y_int)}")

    return X.astype(np.float32), y_int.astype(np.int64)

# Trains EEGNet on a single subject and evaluates performance
# Returns the accuracy of the model and the confusion matrix
def train_eegnet(subject, n_epochs = 40, batch_size = 32, lr = 1e-3):
    print(f"Training EEGNet on subject {subject}...")

    X, y = load_subject(subject)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, shuffle = True, stratify = y
    )

    # Convert to PyTorch tensors
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGNet(
        n_chans = X.shape[1],
        n_outputs = 2,
        n_times = X.shape[2]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    # Train model
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_data)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predicted = preds.argmax(dim = 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

            preds_all.extend(predicted.cpu().numpy())
            labels_all.extend(yb.cpu().numpy())
    
    accuracy = correct / total
    print(f"Subject {subject} - Test Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(labels_all, preds_all)
    print("Confusion Matrix:")
    print(cm)

    return accuracy, cm

subjects = [1, 2, 3]
accuracys = []
confusion_matrices = []
generated_figs = []

if __name__ == "__main__":
    for subject in subjects:
        accuracy, cm = train_eegnet(subject = subject)
        accuracys.append(accuracy)
        confusion_matrices.append((subject, cm))
    
    print("Accuracies:", accuracys)
    print("Average Accuracy:", np.mean(accuracys))

    for subject, matrix in confusion_matrices:
        plt.imshow(matrix, cmap = "Blues")
        plt.title(f"Single Subject Confusion Matrix - Subject {subject}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.show(block = False)
        plt.savefig(f"figs/week_4/single_subject_eegnet_cm_subject_{subject}.png")
        generated_figs.append(f"single_subject_eegnet_cm_subject_{subject}.png")
        with open("figs/week_4/single_subject_generated_figs.txt", "w") as f:
            f.write("\n".join(generated_figs))
