################################################################################
# Implements a CSP + LDA pipeline using the MOABB dataset
################################################################################

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

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
    print(f"Building pipeline on subject {subject}...")
    X, y, meta = paradigm.get_data(dataset = dataset, subjects = [subject])

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    csp = CSP(n_components = 6, reg = None, log = True, norm_trace = False)
    lda = LinearDiscriminantAnalysis()

    clf = Pipeline([
        ('CSP', csp),
        ('LDA', lda)
    ])

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    scores = cross_val_score(clf, X, y, cv = cv)

    print(f"CV scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2f}")

    scores_all.append(scores.mean())

print(f"Accuracies: {scores_all}")
print(f"Average accuracy: {np.mean(scores_all):.2f}")
    
