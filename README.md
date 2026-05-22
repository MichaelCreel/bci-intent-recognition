# BCI Intent Recognition

This project is intended to utilize an AI model for determining the confidence of an intention. The model will be trained using EEG data and designed to determine whether an action performed by a user is deterministic or hesitant. This result will modify drone movement based on the confidence determined by the model, making sure hesitant actions are performed safely and smoothly while deterministic actions are performed as the user wishes.

## Dependencies

- Python
- MNE-Python
- Braindecode
- MOABB

PyTorch will be installed with Braindecode, but installing the CUDA based PyTorch (if not automatically installed) will allow PyTorch to be accelerated using your discrete video card if available.