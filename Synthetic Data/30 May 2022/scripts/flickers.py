## flickers.py
## Power Quality Disturbance: Flickers
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = (1 + a(f) * sin (b * w * t)) * sin (w * t)
#   Parameters: 0.1 <= a(f) <= 0.2, 5 Hz <= b <= 20 Hz

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Generate another sine wave at the flicker frequency.
# Multiply the flicker wave with the flicker magnitude.
# Multiply the two waves together to produce the flickers signals.

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
# Define the flickers parameters: 
flicker_alpha = 0.1
flicker_beta = 5
# Modify the output filename: 
output_filename = "flickers.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)

# Produce the flickers: 
flicker_wave = 1 + flicker_alpha * np.sin(2 * np.pi * flicker_beta * samples)
output_wave = (sine_wave + noise) * flicker_wave

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("PQDs: Flickers")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# Output the datapoints: 
dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
