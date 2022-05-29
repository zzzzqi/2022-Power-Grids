## harmonics.py
## Power Quality Disturbance: Harmonics
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = a1 * sin (w * t) + a3 * sin (3 * w * t) + a5 * sin (5 * w * t)
#   Parameters: 0.05 <= a3, a5 <= 0.15, and the sum of (a-i)^2 == 1
# Source 2: IEEE Std 1159-2019 - Section 4.4.5.2 Harmonics
#   Harmonics are sinusoidal voltages or currents having frequencies that are integer multiples of the frequency at which the supply system
#   is desgined to operate (termed the fundamental frequency, usually at 50Hz or 60 Hz).

## Implementation note: 
# Define the harmonics parameters.
# Generate three sine waves according to the parameters.
# Concatenate the waves together with the noise to produce the harmoincs wave.

import numpy as np
import pandas as pd
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
# Modify the output filename: 
output_filename = "harmonics.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)

# Produce the harmonics: 
a1_wave = a1_parameter * np.sin(2 * np.pi * WAVE_FREQ * samples)
a3_wave = a3_parameter * np.sin(3 * 2 * np.pi * WAVE_FREQ * samples)
a5_wave = a5_parameter * np.sin(5 * 2 * np.pi * WAVE_FREQ * samples)
output_wave = a1_wave + a3_wave + a5_wave + noise

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("PQDs: Harmonics")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# Output the datapoints: 
dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
