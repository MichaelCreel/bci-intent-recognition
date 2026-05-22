################################################################################
# Tests the MOABB dataset to ensure data is accessible
################################################################################

def test_moabb_dataset():
    try:
        import moabb
    except ImportError:
        print("FAILURE: Unsuccessful MOABB import")
        return False
    
    try:
        from moabb.datasets import BNCI2014_001
    except ImportError:
        print("FAILURE: Unsuccessful BNCI2014_001 import")
        return False
    
    print("Loading BNCI2014_001 dataset...")
    dataset = BNCI2014_001()
    subjects = dataset.subject_list

    print("Dataset loaded successfully.")
    print(f"Number of subjects in the dataset: {len(subjects)}")
    return True

if __name__ == "__main__":
    test_moabb_dataset()
