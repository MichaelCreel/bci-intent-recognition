# BCI Intent Recognition

This project is intended to utilize an AI model for determining the confidence of an intention. The model will be trained using EEG data and designed to determine whether an action performed by a user is deterministic or hesitant. This result will modify drone movement based on the confidence determined by the model, making sure hesitant actions are performed safely and smoothly while deterministic actions are performed as the user wishes.

This is a sophomore research project under [Huixin Zhan](https://github.com/huixin-zhan-ai) through [Zhan Lab](https://zhan-lab-ai.github.io/) at New Mexico Tech.

## Dependencies

- Python
- MNE-Python
- Braindecode
- MOABB
- Numpy
- Spicy
- Matplotlib
- Pandas
- Scikit-learn
- AutoReject
- Torch
- Transformers

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

- `models_test.py` imports models from the models folder, tests them, and trains them on sample motor imagery subjects to compare their performance.
    ```bash
    python3 ./tests/models_test.py
    ```

## Notebooks

Notebooks are scripts that practice using datasets, models, and pipelines to learn valuable information about the data and models. This information will be used to determine and design the best final model and pipeline for the project.

## Notes

All commands are written in Bash and should work natively on Linux and MacOS.

```bash
python3 ./example_script.py
```

Windows users will need to use a bash terminal emulator such as [Git Bash](https://git-scm.com/install/windows) (automatically installed with Git for Windows) to run the commands. The commands may also need to be modified to run on windows (e.g. using `python` instead of `python3`).

```bash
python ./example_script.py
```

An alias can be added to the bash emulator to substitute `python3` for `python` to require less modification of the commands.

```bash
alias python3=python
```
