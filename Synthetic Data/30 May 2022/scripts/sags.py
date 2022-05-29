## sags.py
## Power Quality Disturbance: Sags
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = (1 - a(u(t - t1) - u(t - t2))) * sin(w * t)
#   Parameters: 0.1 <= a <= 0.9 && T <= t2 - t1 <= 9T
# Source 2: IEEE Std 1159-2019 - Section 4.4.2.2 Sags(dips)
#   A sag is a decrease in rms voltage to between 0.1 pu and 0.9 pu for durations from 0.5 cycles to 1 min.

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Select a number of values in the wave with the desired time period and
# modify their values to produce the sags.
# Can vary the sag-related parameters (voltage decrease and time duration)
# to create different samples of sags.

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
# Define the sag parameters:
sag_duration_in_cycles = 20
sag_magnitude = 0.1
# Modify the output filename: 
output_filename = "sags.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
output_wave = sine_wave + noise

# Produce the sags: 
sag_duration_in_samples = int(sag_duration_in_cycles * SAMPLES_PER_CYCLE)
sag_starting_index = int(randrange(samples.size - sag_duration_in_samples))
for i in range(sag_starting_index, sag_starting_index + sag_duration_in_samples):
    output_wave[i] *= sag_magnitude

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("PQDs: Sags")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# Output the datapoints: 
dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
