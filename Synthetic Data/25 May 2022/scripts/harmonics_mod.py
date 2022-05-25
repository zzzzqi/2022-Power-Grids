import numpy as np
import pandas as pd
from random import randrange
from matplotlib import pyplot as plt

## Modify filename parameters
# input_filename = "normal_signals.dat"
output_filename = "harmonics_signals.csv"

# input_file = np.fromfile(open(input_filename), dtype=np.float32)

# Power Quality Disturbance: harmonics
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network
# Table 1: 
# Formula: v(t) = a1 * sin (w * t) + a3 * sin (3 * w * t) + a5 * sin (5 * w * t)
# Parameters: 0.05 <= a3, a5 <= 0.15, the sum of (a-i)^2 == 1

# Source 2: IEEE Std 1159-2019
# Section 4.4.5.2 Harmonics
# Harmonics are sinusoidal voltages or currents having frequencies that are integer multiples of the frequency at which the supply system
# is desgined to operate (termed the fundamental frequency, usually at 50Hz or 60 Hz).

# Implementation note:
# Can vary the two parameters (a3 and a5, while a1 is the dependent) to create different samples of harmonics
# input_items is a structure of 1) an array of values, 2) the datatype
# Concatenate three sine waves together with the different a-i parameters in order to generate the harmonics signals

## Define the waveform parameters as constants
FREQ = 60
SAMPLES_PER_CYCLE = 256
AMPLITUDE = 1
TIME_DURATION_IN_MS = 1000
SAMPLES_PER_SIGNAL = FREQ * SAMPLES_PER_CYCLE

## Define the input parameters for the harmonics
a3_parameter = 0.13                                                     ## TO MODIFY
a5_parameter = 0.13                                                     ## TO MODIFY
a1_parameter = (1 - a3_parameter ** 2 - a5_parameter ** 2) ** 0.5       ## DEPENDENT

samples = np.arange(SAMPLES_PER_SIGNAL) # the points on the x axis for plotting
a1_wave = a1_parameter * np.sin(2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))
a3_wave = a3_parameter * np.sin(3 * 2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))
a5_wave = a5_parameter * np.sin(5 * 2 * np.pi * FREQ * (samples / SAMPLES_PER_SIGNAL))
output_wave = a1_wave + a3_wave + a5_wave

dataframe = pd.DataFrame(output_wave)
dataframe.to_csv(output_filename, index=True)
