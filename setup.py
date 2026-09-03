import os
import sys
import subprocess
import venv

def get_venv_path(env_dir):
    if sys.platform == "win32":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")

def run_command(command):
    subprocess.check_call(command)

def main():
    print(f"Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"Working Directory: {os.getcwd()}")

    # Create virtual environment
    env_dir = os.path.abspath("bci-env")
    print(f"Virtual Environment Directory: {env_dir}")

    if not os.path.exists(env_dir):
        venv.EnvBuilder(with_pip=True).create(env_dir)
        print(f"Created Virtual Environment at: {env_dir}")
    else:
        print(f"Virtual Environment at {env_dir} already exists.")

    # Upgrade pip
    venv_python = get_venv_path(env_dir)

    run_command([venv_python, "-m", "pip", "install", "--upgrade", "pip"])

    # Install packages
    packages = [
        "mne",
        "braindecode",
        "moabb",
        "numpy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "torch",
        "autoreject",
        "transformers",
    ]

    for package in packages:
        print(f"Installing Package: {package}")
        run_command([venv_python, "-m", "pip", "install", package])

    print("Setup Completed.")

if __name__ == "__main__":
    main()
