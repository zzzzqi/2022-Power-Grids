import numpy as np
import pandas as pd
from random import randrange
from matplotlib import pyplot as plt

## Modify filename parameters
output_filename = "../output_csv_files/normal_signals.csv"

# Implementation note:
# Generate a normal sine wave with the input waveform parameters

## Define the waveform parameters as constants
FREQ = 60
SAMPLES_PER_CYCLE = 256
AMPLITUDE = 1
TIME_DURATION_IN_MS = 1000
SAMPLES_PER_SIGNAL = FREQ * SAMPLES_PER_CYCLE

samples = np.arange(SAMPLES_PER_SIGNAL) # the points on the x axis for plotting
output_wave = np.sin(2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))

x = np.arange(0, len(output_wave))
plt.title("Normal signals")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
