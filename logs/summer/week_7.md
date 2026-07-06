# Week 7
## Summer

I began creating the safety layer for the BCI system. This safety layer uses the determined confidence, quality score, and temporal stability to determine whether a command should be executed or not. The safety layer code includes the ability to use either CSP + LDA or EEGNet for the intent recognition model, allowing both to be tested.

**CSP + LDA Test Output**  
Mean Intent Confidence: 0.948  
Mean Quality Score: 0.997  
Mean Stability: 0.922  
Mean Safety Score: 0.893  
Safety Score Std Dev: 0.212  
Execute Percentage: 82.29%  

**EEGNet Test Output:**  
Mean Intent Confidence: 0.921  
Mean Quality Score: 0.997  
Mean Stability: 0.905  
Mean Safety Score: 0.857  
Safety Score Std Dev: 0.202  
Execute Percentage: 80.21%

Despite EEGNet's overconfidence from Week 5, CSP + LDA had a higher percentage of commands executed in minimum mode.

### Files

- notebooks/summer/week_7/safety_signal.py

- models/csp_lda.py
- models/eegnet.py
- models/epoch_scorer.py