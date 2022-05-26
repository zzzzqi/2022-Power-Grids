import numpy as np
import pandas as pd
from random import randrange
from matplotlib import pyplot as plt

## Modify filename parameters
# input_filename = "normal_signals.dat"
output_filename = "sags_signals.csv"

# input_file = np.fromfile(open(input_filename), dtype=np.float32)

# Power Quality Disturbance: Sag
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network
# Table 1: 
# Formula: v(t) = (1 - a(u(t - t1) - u(t - t2))) * sin(w * t)
# Parameters: 0.1 <= a <= 0.9 && T <= t2 - t1 <= 9T

# Source 2: IEEE Std 1159-2019
# Section 4.4.2.2 Sags(dips)
# A sag is a decrease in rms voltage to between 0.1 pu and 0.9 pu for durations from 0.5 cycles to 1 min.

# Implementation note:
# Can vary the two parameters (voltage decrease and time duration) to create different samples of sags
# input_items is a structure of 1) an array of values, 2) the datatype
# Select a number of values in the array according to the desired time period and modify their values to do the sags
# The set waveform parameters: 60 cycles per 1,000ms, 256 samples per cycle, amplitude at 1 pu ==> a total of 15,360 samples over 1,000ms
# Hence, the duration of 1 sample takes 0.0651041666ms
# 0.5 cycles are equivalent to 8.3333333ms ==> 127.999999999999 samples

## Define the waveform parameters as constants
FREQ = 60
SAMPLES_PER_CYCLE = 256
AMPLITUDE = 1
TIME_DURATION_IN_MS = 1000
SAMPLES_PER_SIGNAL = FREQ * SAMPLES_PER_CYCLE

## Define the input parameters for the sags
sag_duration_in_cycles = 20                 ## TO MODIFY
sag_magnitude = 0.1                         ## TO MODIFY

sag_duration_in_num_of_samples = int(SAMPLES_PER_SIGNAL / FREQ * sag_duration_in_cycles)
sag_starting_index = int(randrange(SAMPLES_PER_SIGNAL - sag_duration_in_num_of_samples))

samples = np.arange(SAMPLES_PER_SIGNAL) # the points on the x axis for plotting
output_wave = np.sin(2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))

for i in range(sag_starting_index, sag_starting_index + sag_duration_in_num_of_samples):
    output_wave[i] *= sag_magnitude

x = np.arange(0, len(output_wave))
plt.title("PQDs: sags")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
