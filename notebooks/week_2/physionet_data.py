################################################################################
# Loads PhysioNet data for experimentation
################################################################################

from moabb.datasets import PhysionetMI
import mne

dataset = PhysionetMI()

# Load data for subject 1
data = dataset.get_data(subjects=[1])

print(f"Keys: {data.keys()}")
print(f"Subject 1 type: {type(data[1])}")

subject_data = data[1]

# Raw data
raw = subject_data['0']['0']
raw.plot()

# Filtered Data
raw_filtered = raw.copy().filter(8, 30)
raw_filtered.plot()

print("\n\n")

# Extract events
events, event_id = mne.events_from_annotations(raw)
print(event_id)
print(events[:10])

# Create epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id = event_id,
    tmin = 0,
    tmax = 4,
    baseline = None,
    preload = True
)

# Plot epochs
epochs.plot()
    
# Plot epoch for C3
epochs.plot(picks="C3")

# Plot PSD
raw.plot_psd(fmax = 40)

# Perform high pass on PSD and plot
raw_hp = raw.copy().filter(1., None)
raw_hp.plot_psd(fmax = 40)

# Retrieve PSD for left and right hands
epochs_left = epochs['left_hand']
epochs_right = epochs['right_hand']

# Plot C3 PSDs for left and right hand
epochs_left.plot_psd(picks="C3", fmax=40)
epochs_right.plot_psd(picks="C3", fmax=40)

# Plot C4 PSDs for left and right hand
epochs_left.plot_psd(picks="C4", fmax=40)
epochs_right.plot_psd(picks="C4", fmax=40)

mne.viz.utils.plt_show()
