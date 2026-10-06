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

    min_python = (3, 11)
    ins_python = sys.version_info[:2]

    print(f"Found Python {ins_python[0]}.{ins_python[1]}")

    if ins_python < min_python:
        print(f"Python {min_python[0]}.{min_python[1]} or higher is required.")
        sys.exit(1)

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

    requirements_path = os.path.abspath("requirements.txt")

    print(f"Installing packages listed in {requirements_path}")
    run_command([venv_python, "-m", "pip", "install", "-r", requirements_path])

    print("Setup Completed.")

if __name__ == "__main__":
    main()
