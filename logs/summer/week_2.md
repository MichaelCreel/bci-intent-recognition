# Week 2
## Summer

I loaded one subject from PhysioNet movement dataset and plotted the data in mutliple different ways. I extracted events from the data and separated epochs using MNE. I applied filters to the data to extract specific information. I attempted to find Mu and Beta rhythms in plotted EEG data, though finding peaks in the correctly correlating frequency regions proved to be pretty impossible as the peaks are extremely small. The Mu Rhythm appeared to peak around 12-13 Hz and the Beta Rhythym appeared to peak around 22-23 Hz, though the Beta Rhythm appeared to be only one reading.

- EEG Data: Readings of brain activity. Readings can correlate to voluntary movement, involuntary movement, imaginations, or any other sort of brain activity. It is mainly a general reading of electrical signals in the brain. Completely unintelligible without filtering, epoching, and processing. Artificial intelligence clearly has a use-case for understanding here.

- Epochs: Small segments of EEG data that focus on performing one action. Instead of displaying/reading/understanding EEG data continuously, each trial is processed separately.

- Mu Rhythm: Electrical activity patterns that come from brain areas controlling voluntary movement. These are most noticable when the body is at rest. When at rest, the neuron firing is synchronized. When moving, the neuron firing is desynchronized. This means that this band is inversely correlated to voluntary movement/imagined movement. [Source](https://en.wikipedia.org/wiki/Mu_wave)

- Beta Rhythm: This Rhythm is a sign of stability. When the body is resting, the rhythm is high. When the body is moving or imagining movement, the rhythm is low. When going from resting to moving, the rhythm goes from high to low. When going from moving to resting, the rhythm goes from low to high. So, a change in the Beta Rhythm means a change in movement. [Source](https://en.wikipedia.org/wiki/Beta_wave)

### Files

- notebooks/summer/week_2/physionet_data.py
