## normal_signals.py
## Implementation note: 
# Generate a normal sine wave with the input waveform parameters.
# Generate the corresponding number of samples from a normal/ Gaussian distribution.
# Concatenate the two sample waves to produce the output wave.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Define the waveform parameters as constants: 
WAVE_AMPLITUDE = 1
WAVE_FREQ = 60                                                      # 60 cycles per second
SAMPLES_PER_CYCLE = 256                                             # 256 samples per cycle
SIGNAL_DURATION_IN_CYCLES = 10                                      # signal has 10 cycles
SIGNAL_DURATION_IN_SEC = SIGNAL_DURATION_IN_CYCLES / WAVE_FREQ      # signal lasts 10 / 60 seconds
# Define the noise parameters as constants: 
NOISE_SD = 0.005
# Modify the output filename: 
output_filename = "normal_signals.csv"

# Generate the output wave: 
samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / SAMPLES_PER_CYCLE / SIGNAL_DURATION_IN_CYCLES / WAVE_FREQ * 10)
sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
output_wave = sine_wave + noise

# Plot the output wave: 
x = np.arange(0, len(output_wave))
plt.title("Normal electrical signals")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

# # Output the datapoints: 
# dataframe = pd.DataFrame(output_wave)
# dataframe.to_csv(output_filename, index=True)
