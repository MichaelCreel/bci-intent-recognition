# Week 9
## Summer

I created a script to evaluate the performance of all the models using the motor imagery dataset. The script evaluates the models on accuracy and confidence, expected calibration error (ECE), and maximum calibration error (MCE). The script generates reliability diagrams and confidence and accuracy histograms for each model. This allows visual comparison between models that shows how accurate each model is.

The results show that CSP + LDA tends to be underconfident by highly accurate for confidences below 0.55, but is nearly perfect for confidences above 0.55. EEGNet tends to be underconfident at extremely low confidences (below 0.20), but is pretty accurate for those confidences. For confidences above 0.20, EEGNet tends to stay around 0.50 accuracy, being underconfident until around 0.50 confidence and overconfident after. BIOT tends to be mixed with a generally high variance, but is generally 0.50 accurate for all confidences, making it underconfident for confidences below 0.50 and overconfident for confidences above 0.50.

***CSP + LDA***
Accuracy: 0.6458
Mean Confidence: 0.6345
Std Confidence: 0.1838
Expected Calibration Error (ECE): 0.1954
Maximum Calibration Error (MCE): 0.7292
Safety Threshold: 0.75
Acceptance Rate: 0.3264
Reject Rate: 0.6736
Accuracy Above Threshold: 0.7660

****EEGNet****
Accuracy: 0.5938
Mean Confidence: 0.7193
Std Confidence: 0.1714
Expected Calibration Error (ECE): 0.2463
Maximum Calibration Error (MCE): 0.8163
Safety Threshold: 0.75
Acceptance Rate: 0.4792
Reject Rate: 0.5208
Accuracy Above Threshold: 0.7101

***BIOT***
Accuracy: 0.5000
Mean Confidence: 0.4988
Std Confidence: 0.3973
Expected Calibration Error (ECE): 0.3840
Maximum Calibration Error (MCE): 0.4974
Safety Threshold: 0.75
Acceptance Rate: 0.3958
Reject Rate: 0.6042
Accuracy Above Threshold: 0.5175


### Files

- notebooks/summer/week_9/evaluation.py