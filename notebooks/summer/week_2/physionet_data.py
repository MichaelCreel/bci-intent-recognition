################################################################################
# Loads PhysioNet data for experimentation
################################################################################

from moabb.datasets import PhysionetMI
import mne

generated_figs = []

# Save figures to figs/week_2/
def save_figure(fig, filename):
    fig.savefig(f"figs/week_2/{filename}.png")
    generated_figs.append(f"{filename}.png")
    with open("figs/week_2/generated_figs.txt", "w") as f:
        f.write("\n".join(generated_figs))

dataset = PhysionetMI()

# Load data for subject 1
data = dataset.get_data(subjects=[1])

print(f"Keys: {data.keys()}")
print(f"Subject 1 type: {type(data[1])}")

subject_data = data[1]

# Raw data from run 0, session 0
raw = subject_data['0']['0']
fig = raw.plot()
fig.suptitle("Raw EEG Data - Subject 1 - Run 0 - Session 0")
save_figure(fig, "Raw_EEG_Data_Subject_1_Run_0_Session_0")

# Filtered Data between 8 and 30 Hz
raw_filtered = raw.copy().filter(8, 30)
fig = raw_filtered.plot()
fig.suptitle("Filtered EEG Data - Subject 1 - Run 0 - Session 0")
save_figure(fig, "Filtered_EEG_Data_Subject_1_Run_0_Session_0")

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
fig = epochs.plot()
fig.suptitle("Epochs - Subject 1 - Run 0 - Session 0")
save_figure(fig, "Epochs_Subject_1_Run_0_Session_0")

# Plot epoch for C3
fig = epochs.plot(picks="C3")
fig.suptitle("Epochs - Subject 1 - Run 0 - Session 0 - C3")
save_figure(fig, "Epochs_Subject_1_Run_0_Session_0_C3")

# Plot PSD
fig = raw.plot_psd(fmax = 40)
fig.suptitle("PSD - Subject 1 - Run 0 - Session 0")
save_figure(fig, "PSD_Subject_1_Run_0_Session_0")

# Perform high pass on PSD and plot
raw_hp = raw.copy().filter(1., None)
fig = raw_hp.plot_psd(fmax = 40)
fig.suptitle("High Pass PSD - Subject 1 - Run 0 - Session 0")
save_figure(fig, "High_Pass_PSD_Subject_1_Run_0_Session_0")

# Retrieve PSD for left and right hands
epochs_left = epochs['left_hand']
epochs_right = epochs['right_hand']

# Plot C3 PSDs for left and right hand
fig = epochs_left.plot_psd(picks="C3", fmax=40)
fig.suptitle("C3 PSD - Left Hand - Subject 1 - Run 0 - Session 0")
save_figure(fig, "C3_PSD_Left_Hand_Subject_1_Run_0_Session_0")
fig = epochs_right.plot_psd(picks="C3", fmax=40)
fig.suptitle("C3 PSD - Right Hand - Subject 1 - Run 0 - Session 0")
save_figure(fig, "C3_PSD_Right_Hand_Subject_1_Run_0_Session_0")

# Plot C4 PSDs for left and right hand
fig = epochs_left.plot_psd(picks="C4", fmax=40)
fig.suptitle("C4 PSD - Left Hand - Subject 1 - Run 0 - Session 0")
save_figure(fig, "C4_PSD_Left_Hand_Subject_1_Run_0_Session_0")
fig = epochs_right.plot_psd(picks="C4", fmax=40)
fig.suptitle("C4 PSD - Right Hand - Subject 1 - Run 0 - Session 0")
save_figure(fig, "C4_PSD_Right_Hand_Subject_1_Run_0_Session_0")

mne.viz.utils.plt_show()
