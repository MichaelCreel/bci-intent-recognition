# Week 8
## Summer

I compared the performance of BIOT and EEGNet. BIOT was first tried with the backbone frozen, preventing the model from training and outputting guessed values. The model backbone was then unfrozen, allowing the model to train and output values based on the training data, resulting in slightly more accurate guesses.

BIOT and EEGNet were then compared. With a frozen backbone, BIOT was less accurate than EEGNet and was extremely overconfident. With an unfrozen backbone, BIOT was still less accurate than EEGNet, but was more accurate than with a frozen backbone and was less confident in its guesses. The confidence of BIOT's guesses are highly varied, showing that EEGNet is more accurate and consistent in its guesses. These results show that EEGNet outperforms BIOT on smaller training datasets. Three trials are shown below, all with an unfrozen backbone.

***T1:***
EEGNet vs BIOT Accuracy: 0.5451 vs 0.4896
EEGNet vs BIOT Mean Confidence: 0.7913 vs 0.3660
EEGNet vs BIOT Std Confidence: 0.1400 vs 0.3344

***T2:***
EEGNet vs BIOT Accuracy: 0.5694 vs 0.4236
EEGNet vs BIOT Mean Confidence: 0.7786 vs 0.5116
EEGNet vs BIOT Std Confidence: 0.1813 vs 0.3746

***T3:***
EEGNet vs BIOT Accuracy: 0.6632 vs 0.5312
EEGNet vs BIOT Mean Confidence: 0.7121 vs 0.6337
EEGNet vs BIOT Std Confidence: 0.2310 vs 0.3603

### Files

- notebooks/summer/week_8/pretrained_model_comparison.py