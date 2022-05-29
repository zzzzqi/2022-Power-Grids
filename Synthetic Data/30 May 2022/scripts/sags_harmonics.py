## sags_harmonics.py
## Power Quality Disturbance: Sags and harmonics
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = (1 - a(u(t - t1) - u(t - t2))) * (a1 * sin (w * t) + a3 * sin (3 * w * t) + a5 * sin (5 * w * t))
#   Parameters for sags: 0.1 <= a <= 0.9 && T <= t2 - t1 <= 9T
#   Parameters for harmonics: 0.05 <= a3, a5 <= 0.15, the sum of (a-i)^2 == 1
# Source 2: IEEE Std 1159-2019 - Section 4.4.2.2 Sags(dips)
#   A sag is a decrease in rms voltage to between 0.1 pu and 0.9 pu for durations from 0.5 cycles to 1 min.
# Source 2 (cont'd): IEEE Std 1159-2019 - Section 4.4.5.2 Harmonics
#   Harmonics are sinusoidal voltages or currents having frequencies that are integer multiples of the frequency at which the supply system
#   is desgined to operate (termed the fundamental frequency, usually at 50Hz or 60 Hz).

## Implementation note: 
# Generate three sine waves according to the parameters.
# Concatenate the waves together with the noise to produce the harmoincs wave.
# Select a number of values in the wave with the desired time period and
# modify their values to produce the sags.
# Can vary the harmonics-related or sag-related parameters (voltage decrease and time duration)
# to create different samples of sags and harmonics.

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
# Define the harmonics parameters: 
a3_parameter = 0.15
a5_parameter = 0.15
a1_parameter = (1 - a3_parameter ** 2 - a5_parameter ** 2) ** 0.5
# Define the sag parameters:
sag_duration_in_cycles = 20
sag_magnitude = 0.1
# Modify the output filename: 
output_filename = "sags_harmonics.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)

# Produce the harmonics: 
a1_wave = a1_parameter * np.sin(2 * np.pi * WAVE_FREQ * samples)
a3_wave = a3_parameter * np.sin(3 * 2 * np.pi * WAVE_FREQ * samples)
a5_wave = a5_parameter * np.sin(5 * 2 * np.pi * WAVE_FREQ * samples)
output_wave = a1_wave + a3_wave + a5_wave + noise

# Produce the sags: 
sag_duration_in_samples = int(sag_duration_in_cycles * SAMPLES_PER_CYCLE)
sag_starting_index = int(randrange(samples.size - sag_duration_in_samples))
for i in range(sag_starting_index, sag_starting_index + sag_duration_in_samples):
    output_wave[i] *= sag_magnitude

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("PQDs: Sags and harmonics")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# Output the datapoints: 
dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
