## spikes.py
## Power Quality Disturbance: Spikes
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = sin (w * t) + sign(sin (w * t)) * SUM(n:=0->9) (K * (u * (t - (t1 + 0.02n)) - u * (t - (t2 + 0.02n))))
#   Parameters: 0.1 <= K <= 0.4, 0T <= t1, t2 <= 0.5T, 0.01T <= t2 -t1 <= 0.05T

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Randomise the locations/ points of occurrence of the spikes.
# At these points, concatenate the normal wave with the product of the sign of the normal wave element and the spike magnitude.

import numpy as np
import pandas as pd
from random import randrange
from matplotlib import pyplot as plt

# Define the waveform parameters as constants: 
WAVE_AMPLITUDE = 1
WAVE_FREQ = 60
SAMPLES_PER_CYCLE = 256
SIGNAL_DURATION_IN_SEC = 1
# Define the noise parameters as constants: 
NOISE_SD = 0.05
# Define the spike parameters: 
spike_magnitude = 0.4
spike_duration_in_cycles = 0.5

# Modify the output filename: 
output_filename = "spikes.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
output_wave = sine_wave + noise

# Produce the spikes: 
spike_duration_in_num_of_samples = int(spike_duration_in_cycles * SAMPLES_PER_CYCLE)
spike_starting_index = int(randrange(samples.size - spike_duration_in_num_of_samples))
for i in range(spike_starting_index, spike_starting_index + spike_duration_in_num_of_samples):
    output_wave[i] += np.sign(output_wave[i]) * spike_magnitude

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("PQDs: Spikes")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# Output the datapoints: 
dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
