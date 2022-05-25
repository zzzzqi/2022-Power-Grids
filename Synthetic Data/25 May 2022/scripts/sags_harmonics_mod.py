import numpy as np
import pandas as pd
from random import randrange
from matplotlib import pyplot as plt

## Modify filename parameters
# input_filename = "normal_signals.dat"
output_filename = "sags_harmonics_signals.csv"

# input_file = np.fromfile(open(input_filename), dtype=np.float32)

# Power Quality Disturbance: sags and harmonics
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network
# Table 1: 
# Formula: v(t) = (1 - a(u(t - t1) - u(t - t2))) * (a1 * sin (w * t) + a3 * sin (3 * w * t) + a5 * sin (5 * w * t))
# Parameters for sags: 0.1 <= a <= 0.9 && T <= t2 - t1 <= 9T
# Parameters for harmonics: 0.05 <= a3, a5 <= 0.15, the sum of (a-i)^2 == 1

# Source 2: IEEE Std 1159-2019
# Section 4.4.2.2 Sags(dips)
# A sag is a decrease in rms voltage to between 0.1 pu and 0.9 pu for durations from 0.5 cycles to 1 min.
# Section 4.4.5.2 Harmonics
# Harmonics are sinusoidal voltages or currents having frequencies that are integer multiples of the frequency at which the supply system
# is desgined to operate (termed the fundamental frequency, usually at 50Hz or 60 Hz).

# Implementation note:
# Can vary the two parameters (voltage decrease and time duration) to create different samples of sags
# Can vary the two parameters (a3 and a5, while a1 is the dependent) to create different samples of harmonics
# input_items is a structure of 1) an array of values, 2) the datatype
# Concatenate three sine waves together with the different a-i parameters in order to generate the harmonics signals
# Then select a number of values in the wave according to the desired time period and modify their values to do the sags

## Define the waveform parameters as constants
FREQ = 60
SAMPLES_PER_CYCLE = 256
AMPLITUDE = 1
TIME_DURATION_IN_MS = 1000
SAMPLES_PER_SIGNAL = FREQ * SAMPLES_PER_CYCLE

## Define the input parameters for the sags
sag_duration_in_cycles = 20                                             ## TO MODIFY
sag_magnitude = 0.5                                                     ## TO MODIFY
sag_duration_in_num_of_samples = int(SAMPLES_PER_SIGNAL / FREQ * sag_duration_in_cycles)        ## DEPENDENT
sag_starting_index = int(randrange(SAMPLES_PER_SIGNAL - sag_duration_in_num_of_samples))        ## DEPENDENT

## Define the input parameters for the harmonics
a3_parameter = 0.15                                                     ## TO MODIFY
a5_parameter = 0.15                                                     ## TO MODIFY
a1_parameter = (1 - a3_parameter ** 2 - a5_parameter ** 2) ** 0.5       ## DEPENDENT

samples = np.arange(SAMPLES_PER_SIGNAL) # the points on the x axis for plotting
a1_wave = a1_parameter * np.sin(2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))
a3_wave = a3_parameter * np.sin(3 * 2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))
a5_wave = a5_parameter * np.sin(5 * 2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))
output_wave = a1_wave + a3_wave + a5_wave

for i in range(sag_starting_index, sag_starting_index + sag_duration_in_num_of_samples):
    output_wave[i] *= sag_magnitude

x = np.arange(0, len(output_wave))
plt.title("PQDs: sags & harmonics")
plt.xlabel("number of signals")
plt.ylabel("amplitude")
plt.plot(x, output_wave)
plt.show()

dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
