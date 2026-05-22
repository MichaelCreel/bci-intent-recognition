################################################################################
# Tests the environment to ensure all packages are installed and accessible
################################################################################

# Tests that each package can be imported successfully
def test_environment():
    mne_success = True
    braindecode_success = True
    moabb_success = True

    try:
        import mne
    except ImportError:
        mne_success = False
        print("FAILURE: Unsuccessful MNE import")

    try:
        import braindecode
    except ImportError:
        braindecode_success = False
        print("FAILURE: Unsuccessful Braindecode import")

    try:
        import moabb
    except ImportError:
        moabb_success = False
        print("FAILURE: Unsuccessful MOABB import")
    if (mne_success and braindecode_success and moabb_success):
        print("All packages imported successfully.")
    else:
        print("Some packages failed to import:")
        if not mne_success:
            print("  - MNE")
        if not braindecode_success:
            print("  - Braindecode")
        if not moabb_success:
            print("  - MOABB")
        print("Ensure all packages are installed and accessible.")
        print("    - Ensure the environment is activated")
        print("    - Ensure that all dependencies are installed")

if __name__ == "__main__":
    test_environment()
    