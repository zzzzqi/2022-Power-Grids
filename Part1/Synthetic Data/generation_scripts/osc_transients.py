## osc_transients.py
## Power Quality Disturbance: Oscillatory Transients
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = sin * w * t + a * (e ^ (-(t - t1) / T0)) * sin * w(n) * (t - t1) * (u(t2) - u(t1))
#   Parameters: 0.1 <= a <= 0.8, 0.5T <= t2 - t1 <= 3T, 8ms <= T0 <= 40ms, 300Hz <= fn <= 900Hz
# Source 2: IEEE Std 1159-2019 - Section 4.4.1.2 Oscillatory transient
#   An oscillatory transient is a sudden, non-power frequency change in the steady-state conditon of voltage, current, or both,
#   that includes both positive and negative polarity values.
#   An oscillatory transient consists of a voltage or current whose instantaneous value changes polarity rapidly multiple times and
#   normally decaying within a fundamental-frequency cycle.
#   High-frequency oscillatory transients:
#       A primary frequency component greater than 500 kHz and
#       a typical duration measured in microseconds (or several cycles of the principal frequency)
#   Medium-frequency oscillatory transients:
#       A primary frequency component bewteen 5 kHz and 500 kHz and
#       a typical duration measured in tens of microseconds (or several cycles of the principal frequency)
#   Low-frequency oscillatory transients:
#       A primary frequency component less than 5 kHz and
#       a typical duration from 0.3ms to 50ms

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Generate another sine wave at the input transient frequency.
# Produce the transient term with the input transient magnitude and duration.
# Randomise the locations/ points of occurrence of the transients.
# In these locations, concatenate the transients to the normal wave to produce the oscillatory transients.

import numpy as np
import pandas as pd
import csv
import random
from random import randrange
from matplotlib import pyplot as plt

# Define the waveform parameters as constants: 
WAVE_AMPLITUDE = 1
WAVE_FREQ = 60                                                      # 60 cycles per second
SAMPLES_PER_CYCLE = 256                                             # 256 samples per cycle
SIGNAL_DURATION_IN_CYCLES = 10                                      # signal has 10 cycles
SIGNAL_DURATION_IN_SEC = SIGNAL_DURATION_IN_CYCLES / WAVE_FREQ      # signal lasts 10 / 60 seconds
# Define the noise parameters as constants: 
NOISE_SD = 0.005

# Define the function to generate osc transients
def generate_signals(n):
    for i in range(n):
        # Define the transient parameters: 
        transient_alpha = random.choice(range(10, 85, 5)) / 100
        transient_duration_in_cycles = random.choice(range(5, 35, 5)) / 10
        transient_duration_in_ms = random.choice(range(8, 48, 8))
        transient_frequency = random.choice(range(300, 1000, 100))
        # Modify the output filename: 
        output_filename = "osc_transients_sample0" + str(i) + ".csv"

        # Generate the output wave: 
        samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / SAMPLES_PER_CYCLE / SIGNAL_DURATION_IN_CYCLES / WAVE_FREQ * 10)
        sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
        noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
        output_wave = sine_wave + noise

        # Produce the oscillatory transients: 
        transient_duration_in_samples = int(transient_duration_in_cycles * SAMPLES_PER_CYCLE)
        transient_starting_index = int(randrange(samples.size - transient_duration_in_samples))
        transient_wave = np.sin(2 * np.pi * transient_frequency * samples)
        transient_term = transient_alpha * (np.e ** (-1 / transient_duration_in_ms))
        for i in range(transient_starting_index, transient_starting_index + transient_duration_in_samples):
            output_wave[i] += transient_wave[i] * transient_term

        # # Plot the output wave: 
        # x = np.arange(0, len(output_wave))
        # plt.title("PQDs: Oscillatory transients")
        # plt.xlabel("number of signals")
        # plt.ylabel("amplitude")
        # plt.plot(x, output_wave)
        # plt.show()

        # Write the datapoints to a CSV file: 
        dataframe = pd.DataFrame(output_wave)
        dataframe.to_csv(output_filename, index=True)

        # Read the CSV file:
        read_file = open(output_filename, 'r')
        reader = csv.reader(read_file)
        file_rows = list(reader)
        read_file.close()
        # Write the timestamps:
        file_rows[0].append("timestamps (in ms)")
        for j in range(1, len(file_rows)):
            file_rows[j].append(SIGNAL_DURATION_IN_SEC * 1000 / WAVE_FREQ / SAMPLES_PER_CYCLE * (j - 1))
        # Write the parameters:
        for k in range(9):
            file_rows[k].append("")
            file_rows[k].append("")
        file_rows[0][1] = "amplitude (in pu)"
        file_rows[0][3] = "parameters"
        file_rows[1][3] = "waveform frequency"
        file_rows[2][3] = "signal duration in ms"
        file_rows[3][3] = "signal sample per cycle"
        file_rows[4][3] = "noise level in sd"
        file_rows[5][3] = "transient alpha"
        file_rows[6][3] = "transient duration in cycles"
        file_rows[7][3] = "transient duration in ms"
        file_rows[8][3] = "transient frequency"
        file_rows[0][4] = "values"
        file_rows[1][4] = WAVE_FREQ
        file_rows[2][4] = SIGNAL_DURATION_IN_SEC * 1000
        file_rows[3][4] = SAMPLES_PER_CYCLE
        file_rows[4][4] = NOISE_SD
        file_rows[5][4] = transient_alpha
        file_rows[6][4] = transient_duration_in_cycles
        file_rows[7][4] = transient_duration_in_ms
        file_rows[8][4] = transient_frequency
        # Edit the CSV file: 
        output_file = open(output_filename, 'w', newline = '')
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(file_rows)
        output_file.close()
