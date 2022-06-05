## interruptions.py
## Power Quality Disturbance: Interruptions
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = (1 - a(u(t - t1) - u(t - t2))) * sin(w * t)
#   Parameters: 0.9 <= a <= 1 && T <= t2 - t1 <= 9T
# Source 2: IEEE Std 1159-2019 - Section 4.4.2.1 Instantaneous, momentary, and temporary interruptions
#   An interruption occurs when the supply voltage or load current decreases to less than 0.1 pu for a period of time not exceeding 1 min.

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Select a number of values in the wave with the desired time period and
# modify their values to produce the interruptions.
# Can vary the interruption-related parameters (voltage decrease and time duration)
# to create different samples of interruptions.

import numpy as np
import pandas as pd
import csv
import random
from random import randrange
from matplotlib import pyplot as plt

# Define the waveform parameters as constants: 
WAVE_AMPLITUDE = 1
WAVE_FREQ = 60
SAMPLES_PER_CYCLE = 256
SIGNAL_DURATION_IN_SEC = 1
# Define the noise parameters as constants: 
NOISE_SD = 0.05
# Define the number of data samples to be created:
NUMBER_OF_SAMPLES = 1000

for i in range(NUMBER_OF_SAMPLES):
    # Define the interruption parameters:
    interruption_duration_in_cycles = random.uniform(0.5, WAVE_FREQ)
    interruption_magnitude = (random.choice(range(0, 10, 1))) / 100
    # Modify the output filename: 
    output_filename = "interruptions_sample0" + str(i) + ".csv"

    # Generate the output wave: 
    samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
    sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
    noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
    output_wave = sine_wave + noise

    # Produce the interruptions: 
    interruption_duration_in_samples = int(interruption_duration_in_cycles * SAMPLES_PER_CYCLE)
    interruption_starting_index = int(randrange(samples.size - interruption_duration_in_samples))
    for ii in range(interruption_starting_index, interruption_starting_index + interruption_duration_in_samples):
        output_wave[ii] *= interruption_magnitude

    # # Plot the output wave: 
    # x = np.arange(0, len(output_wave))
    # plt.title("PQDs: Interruptions")
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
    for k in range(6):
        file_rows[k].append("")
        file_rows[k].append("")
    file_rows[0][1] = "amplitude (in pu)"
    file_rows[0][3] = "parameters"
    file_rows[1][3] = "waveform frequency"
    file_rows[2][3] = "signal duration in ms"
    file_rows[3][3] = "signal sample per cycle"
    file_rows[4][3] = "interruption duration in cycles"
    file_rows[5][3] = "interruption magnitude"
    file_rows[0][4] = "values"
    file_rows[1][4] = WAVE_FREQ
    file_rows[2][4] = SIGNAL_DURATION_IN_SEC * 1000
    file_rows[3][4] = SAMPLES_PER_CYCLE
    file_rows[4][4] = interruption_duration_in_cycles
    file_rows[5][4] = interruption_magnitude
    # Edit the CSV file: 
    output_file = open(output_filename, 'w', newline = '')
    csv_writer = csv.writer(output_file)
    csv_writer.writerows(file_rows)
    output_file.close()
