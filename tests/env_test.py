################################################################################
# Tests the environment to ensure all packages are installed and accessible
################################################################################

import os

def test_environment():
    requirements_path = os.path.abspath("requirements.txt")
    success = True
    packages = open(requirements_path).read().splitlines()
    failed_packages = []

    packages = [package.split("==")[0] for package in packages]

    for package in packages:
        if package == "scikit-learn":
            package = "sklearn"
        try:
            __import__(package)
        except ImportError:
            print(f"Failed to import package {package}.")
            success = False
            failed_packages.append(package)
    
    if success:
        print("All packages imported successfully.")
    else:
        print("Some packages failed to import.\nFailed packages:")
        for package in failed_packages:
            print(f"    - {package}")
        print("Ensure all packages are installed and accessible.")
        print("Ensure the environment is activated.")

if __name__ == "__main__":
    test_environment()
    