# Week 5
## Summer

I reused cross-subject EEGNet model training (since it had a higher average accuracy than single subject EEGNet) and implemented temperature scaling to calibrate the models. I evaluated the models using the expected calibration error (ECE) and creating reliability diagrams. The ECE was about 50/50 for whether temperature scaling reduced or increased the error. The reliability diagrams showed that the temperature scaling generally improved the reliability of the models, though all of the models were overconfident both before and after scaling.

I created CSP + LDA pipelines for multiple subjects and applied temperature scaling to the pipelines. The ECE tended to be reduced by scaling, though not all pipelines had reduced ECE after scaling. The reliability diagrams showed that the pipelines tended to be overconfident before and after scaling, though they were not massively overconfident like the EEGNet models. The scaling generally improved the reliability of the pipelines.

### Files

- notebooks/week_5/eegnet_scaling.py
- notebooks/week_5/csp_lda_scaling.py