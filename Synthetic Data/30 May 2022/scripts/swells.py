## swells.py
## Power Quality Disturbance: Swells
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = (1 + a(u(t - t1) - u(t - t2))) * sin(w * t)
#   Parameters: 0.1 <= a <= 0.8 && T <= t2 - t1 <= 9T
# Source 2: IEEE Std 1159-2019 - Section 4.4.2.3 Swells
#   A swell is a increase in rms voltage to above 1.1 pu for durations from 0.5 cycles to 1 min. Typical magnitudes are between 1.1 and 1.2 pu.

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Select a number of values in the wave with the desired time period and
# modify their values to produce the swells.
# Can vary the swell-related parameters (voltage increase and time duration)
# to create different samples of swells.

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
# Define the swell parameters:
swell_duration_in_cycles = 20
swell_magnitude = 1.2
# Modify the output filename: 
output_filename = "swells.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
output_wave = sine_wave + noise

# Produce the swells: 
swell_duration_in_samples = int(swell_duration_in_cycles * SAMPLES_PER_CYCLE)
swell_starting_index = int(randrange(samples.size - swell_duration_in_samples))
for i in range(swell_starting_index, swell_starting_index + swell_duration_in_samples):
    output_wave[i] *= swell_magnitude

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("PQDs: Swells")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# Output the datapoints: 
dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
