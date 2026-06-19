# Week 4
## Summer

I implemented EEGNet model training for left vs right hand motor imagery. I trained multiple models with some trained on individual subjects and some trained on multiple subjects. For these subjects, I used both imagined and real motor movement data, which had given the highest accuracy in the CSP + LDA pipeline. The average accuracy for the single subject models was 0.46, while the average accuracy for the multi-subject models was 0.60. These accuracies show that the single subject models performed worse than the multi-subject models on average.

### Files

- notebooks/week_4/single_subject_eegnet.py
- notebooks/week_4/cross_subject_eegnet.py