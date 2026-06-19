# Notebooks
## Summer

- `physionet_data.py` loads the PhysioNet movement dataset and plots, filters, and processes data for visualization. It is a clear demonstration of why artificial intelligence should be used to read EEG data, as the datasets and epochs are essentially unintelligable.
    ```bash
    python3 ./notebooks/summer/week_2/physionet_data.py
    ```
- `csp_lda_competition.py` implements a CSP + LDA classification pipeline for left vs right hand motor imagery using raw EEG motor imagery data from Physionet. The program measures the accuracy of the pipeline using the provided data.
    ```bash
    python3 ./notebooks/summer/week_3/csp_lda_competition.py
    ```

- `moabb_csp_lda_competition.py` implements a CSP + LDA classification pipeline for left vs right hand motor imagery using the MOABB dataset. The program determines the accuracy of the pipeline using the provided data.
    ```bash
    python3 ./notebooks/summer/week_3/moabb_csp_lda_competition.py
    ```

- `single_subject_eegnet.py` trains EEGNet models on individual subjects for left vs right hand motor imagery using EEG motor imagery data from the PhysioNet Motor Imagery dataset. The program then evaluates the accuracy of the model and plots the confusion matricies.
    ```bash
    python3 ./notebooks/summer/week_3/single_subject_eegnet.py
    ```

- `cross_subject_eegnet.py` trains EEGNet models on multiple subjects for left vs right hand motor imagery using EEG motor imagery data from the PhysioNet Motor Imagery dataset. The program then evaluates the accuracy of the model and plots the confusion matricies.
    ```bash
    python3 ./notebooks/summer/week_4/cross_subject_eegnet.py
    ```

- `eegnet_scaling.py` trains EEGNet models on multiple subjects in the same way as `cross_subject_eegnet.py`. Applies temperature scaling to the models and evaluates the expected calibration error and plots relability diagrams for before and after temperature scaling.
    ```bash
    python3 ./notebooks/summer/week_5/eegnet_scaling.py
    ```