################################################################################
# Tests the EEGNet model to ensure that it can be imported and used
################################################################################

def test_eegnet():
    try:
        import torch
        from braindecode.models import EEGNet
    except ImportError as e:
        print("FAILURE: Unsuccessful EEGNet or Torch Import")
        return False
    
    print(f"GPU Acceleration Available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dummy_input = torch.randn(2, 22, 1000).to(device)
    
    model = EEGNet(
        n_chans = 22,
        n_outputs = 2,
        n_times = 1000,
    ).to(device)

    print("Forward pass through the model:")
    output = model(dummy_input)
    print("Output shape:", output.shape)
    return True

if __name__ == "__main__":
    test_eegnet()
