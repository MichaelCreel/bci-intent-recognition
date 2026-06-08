# BCI Intent Recognition

This project is intended to utilize an AI model for determining the confidence of an intention. The model will be trained using EEG data and designed to determine whether an action performed by a user is deterministic or hesitant. This result will modify drone movement based on the confidence determined by the model, making sure hesitant actions are performed safely and smoothly while deterministic actions are performed as the user wishes.

This is a sophomore research project under [Huixin Zhan](https://github.com/huixin-zhan-ai) through [Zhan Lab](https://zhan-lab-ai.github.io/) at New Mexico Tech.

## Dependencies

- Python
- MNE-Python
- Braindecode
- MOABB

PyTorch will be installed with Braindecode, but installing the CUDA based PyTorch (if not automatically installed) will allow PyTorch to be accelerated using your discrete video card if available.

## Tests

Tests ensure that your environment is set up and functioning correctly

- `env_test.py` attempts to import the required dependencies to ensure the environment contains the necessary packages. This does not ensure that your environment is running optimally; it only checks if your environment is functioning.
    ```bash
    python3 ./tests/env_test.py
    ```
- `moabb_test.py` attempts to import a dataset from MOABB to ensure that datasets can be loaded and used in the environment.
    ```bash
    python3 ./tests/moabb_test.py
    ```

- `eegnet_test.py` attempts to utilize an EEGNet model to ensure that EEGNet models can be used in the environment.
    ```bash
    python3 ./tests/eegnet_test.py
    ```

## Notebooks

- `physionet_data.py` loads the PhysioNet movement dataset and plots, filters, and processes data for visualization. It is a clear demonstration of why artificial intelligence should be used to read EEG data, as the datasets and epochs are essentially unintelligable.
    ```bash
    python3 ./notebooks/week_2/physionet_data.py
    ```
- `csp_lda_competition.py` implements a CSP + LDA classification pipeline for left vs right hand motor imagery using raw EEG motor imagery data from Physionet. The program measures the accuracy of the pipeline using the provided data.
    ```bash
    python3 ./notebooks/week_3/csp_lda_competition.py
    ```

- `single_subject_eegnet.py` trains EEGNet models on individual subjects for left vs right hand motor imagery using EEG motor imagery data from the PhysioNet Motor Imagery dataset. The program then evaluates the accuracy of the model and plots the confusion matricies.
    ```bash
    python3 ./notebooks/week_3/single_subject_eegnet.py
    ```

- `cross_subject_eegnet.py` trains EEGNet models on multiple subjects for left vs right hand motor imagery using EEG motor imagery data from the PhysioNet Motor Imagery dataset. The program then evaluates the accuracy of the model and plots the confusion matricies.
    ```bash
    python3 ./notebooks/week_4/cross_subject_eegnet.py
    ```