import numpy as np
import pandas as pd
from random import randrange
from matplotlib import pyplot as plt

## Modify filename parameters
# input_filename = "normal_signals.dat"
output_filename = "swells_signals.csv"

# input_file = np.fromfile(open(input_filename), dtype=np.float32)

# Power Quality Disturbance: Swells
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network
# Table 1: 
# Formula: v(t) = (1 + a(u(t - t1) - u(t - t2))) * sin(w * t)
# Parameters: 0.1 <= a <= 0.8 && T <= t2 - t1 <= 9T

# Source 2: IEEE Std 1159-2019
# Section 4.4.2.3 Swells
# A swell is a increase in rms voltage to above 1.1 pu for durations from 0.5 cycles to 1 min. Typical magnitudes are between 1.1 and 1.2 pu.

# Implementation note:
# Can vary the two parameters (voltage increase and time duration) to create different samples of swells
# input_items is a structure of 1) an array of values, 2) the datatype
# Select a number of values in the array according to the desired time period and modify their values to do the swells
# The set waveform parameters: 60 cycles per 1,000ms, 256 samples per cycle, amplitude at 1 pu ==> a total of 15,360 samples over 1,000ms
# Hence, the duration of 1 sample takes 0.0651041666ms
# 0.5 cycles are equivalent to 8.3333333ms ==> 127.999999999999 samples

## Define the waveform parameters as constants
FREQ = 60
SAMPLES_PER_CYCLE = 256
AMPLITUDE = 1
TIME_DURATION_IN_MS = 1000
SAMPLES_PER_SIGNAL = FREQ * SAMPLES_PER_CYCLE

## Define the input parameters for the swells
swell_duration_in_cycles = 20                 ## TO MODIFY
swell_magnitude = 1.2                         ## TO MODIFY

swell_duration_in_num_of_samples = int(SAMPLES_PER_SIGNAL / FREQ * swell_duration_in_cycles)
swell_starting_index = int(randrange(SAMPLES_PER_SIGNAL - swell_duration_in_num_of_samples))

samples = np.arange(SAMPLES_PER_SIGNAL) # the points on the x axis for plotting
output_wave = np.sin(2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))

for i in range(swell_starting_index, swell_starting_index + swell_duration_in_num_of_samples):
    output_wave[i] *= swell_magnitude

x = np.arange(0, len(output_wave))
plt.title("PQDs: swells")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
