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

# Define the function to generate flickers
def generate_signals(n):
    for i in range(n):
        # Define the flickers parameters: 
        flicker_alpha = (random.choice(range(10, 25, 5))) / 100
        flicker_beta = random.choice(range(5, 25, 5))
        # Modify the output filename: 
        output_filename = "flickers_sample0" + str(i) + ".csv"

        # Generate the output wave: 
        samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / SAMPLES_PER_CYCLE / SIGNAL_DURATION_IN_CYCLES / WAVE_FREQ * 10)
        sine_wave = WAVE_AMPLITUDE * np.sin(2 * np.pi * WAVE_FREQ * samples)
        noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)

        # Produce the flickers: 
        flicker_wave = 1 + flicker_alpha * np.sin(2 * np.pi * flicker_beta * samples)
        output_wave = (sine_wave + noise) * flicker_wave

        # # Plot the output wave: 
        # x = np.arange(0, len(output_wave))
        # plt.title("PQDs: Flickers")
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
        for k in range(7):
            file_rows[k].append("")
            file_rows[k].append("")
        file_rows[0][1] = "amplitude (in pu)"
        file_rows[0][3] = "parameters"
        file_rows[1][3] = "waveform frequency"
        file_rows[2][3] = "signal duration in ms"
        file_rows[3][3] = "signal sample per cycle"
        file_rows[4][3] = "noise level in sd"
        file_rows[5][3] = "flicker alpha"
        file_rows[6][3] = "flicker beta"
        file_rows[0][4] = "values"
        file_rows[1][4] = WAVE_FREQ
        file_rows[2][4] = SIGNAL_DURATION_IN_SEC * 1000
        file_rows[3][4] = SAMPLES_PER_CYCLE
        file_rows[4][4] = NOISE_SD
        file_rows[5][4] = flicker_alpha
        file_rows[6][4] = flicker_beta
        # Edit the CSV file: 
        output_file = open(output_filename, 'w', newline = '')
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(file_rows)
        output_file.close()
