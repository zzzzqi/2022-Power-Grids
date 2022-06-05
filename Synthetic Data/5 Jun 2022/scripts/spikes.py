## spikes.py
## Power Quality Disturbance: Spikes
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = sin (w * t) + sign(sin (w * t)) * SUM(n:=0->9) (K * (u * (t - (t1 + 0.02n)) - u * (t - (t2 + 0.02n))))
#   Parameters: 0.1 <= K <= 0.4, 0T <= t1, t2 <= 0.5T, 0.01T <= t2 -t1 <= 0.05T

## Implementation note: 
# Generate a normal sine wave with noise with the input waveform and noise parameters.
# Randomise the locations/ points of occurrence of the spikes.
# At these points, concatenate the normal wave with the product of the sign of the normal wave element and the spike magnitude.

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
    # Define the spike parameters: 
    spike_magnitude = random.choice(range(10, 40, 1)) / 100
    spike_duration_in_cycles = random.choice(range(0, 50, 10)) / 100

    # Modify the output filename: 
    output_filename = "spikes_sample0" + str(i) + ".csv"

    # Generate the output wave: 
    samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / WAVE_FREQ / SAMPLES_PER_CYCLE)
    sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
    noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)
    output_wave = sine_wave + noise

    # Produce the spikes: 
    spike_duration_in_num_of_samples = int(spike_duration_in_cycles * SAMPLES_PER_CYCLE)
    spike_starting_index = int(randrange(samples.size - spike_duration_in_num_of_samples))
    for i in range(spike_starting_index, spike_starting_index + spike_duration_in_num_of_samples):
        output_wave[i] += np.sign(output_wave[i]) * spike_magnitude

    # # Plot the output wave: 
    # x = np.arange(0, len(output_wave))
    # plt.title("PQDs: Spikes")
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
    file_rows[4][3] = "spike magnitude"
    file_rows[5][3] = "spike duration in cycles"
    file_rows[0][4] = "values"
    file_rows[1][4] = WAVE_FREQ
    file_rows[2][4] = SIGNAL_DURATION_IN_SEC * 1000
    file_rows[3][4] = SAMPLES_PER_CYCLE
    file_rows[4][4] = spike_magnitude
    file_rows[5][4] = spike_duration_in_cycles
    # Edit the CSV file: 
    output_file = open(output_filename, 'w', newline = '')
    csv_writer = csv.writer(output_file)
    csv_writer.writerows(file_rows)
    output_file.close()
