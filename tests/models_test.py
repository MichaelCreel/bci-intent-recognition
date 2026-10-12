################################################################################
# Tests the models to ensure that it can be imported and used
################################################################################

from importlib import import_module

def test_models():
    package_success = True
    packages = ["torch", "braindecode", "mne", "sklearn"]
    failed_packages = []

    model_success = True
    # ("Model", "Package"
    models = [
        ("CSP", "mne.decoding"),
        ("LinearDiscriminantAnalysis", "sklearn.discriminant_analysis"),
        ("EEGNet", "braindecode.models"),
        ("InterpolatedBIOT", "braindecode.models")
    ]
    failed_models = []

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Failed to import package {package}.")
            package_success = False
            failed_packages.append(package)

    if package_success:
        for model, package in models:
            try:
                module = import_module(f"{package}")
                getattr(module, model)
            except (ImportError, AttributeError):
                print(f"Failed to import model {model} from package {package}.")
                model_success = False
                failed_models.append(model)

    if package_success and model_success:
        print("All models imported successfully.")
        import torch
        if torch.cuda.is_available():
            print("GPU acceleration available.")
    elif not package_success:
        print("Some dependencies failed to import.")
        for package in failed_packages:
            print(f"    - {package}")
        print("Ensure all packages are installed and accessible.")
        print("Ensure the environment is activated.")
    else:
        print("Some models failed to import.")
        for model in failed_models:
            print(f"    - {model}")

if __name__ == "__main__":
    test_models()
