## interruptions_harmonics.py
## Power Quality Disturbance: Interruptions and harmonics
# Source 1: Classifying Power Quality Disturbances Based on Phase Space Reconstruction and a Convolutional Neural Network - Table 1: 
#   Formula: v(t) = (1 - a(u(t - t1) - u(t - t2))) * (a1 * sin (w * t) + a3 * sin (3 * w * t) + a5 * sin (5 * w * t))
#   Parameters for interruptions: 0.9 <= a <= 1 && T <= t2 - t1 <= 9T
#   Parameters for harmonics: 0.05 <= a3, a5 <= 0.15, the sum of (a-i)^2 == 1
# Source 2: IEEE Std 1159-2019 - Section 4.4.2.1 Instantaneous, momentary, and temporary interruptions
#   An interruption occurs when the supply voltage or load current decreases to less than 0.1 pu for a period of time not exceeding 1 min.
# Source 2 (cont'd): IEEE Std 1159-2019 - Section 4.4.5.2 Harmonics
#   Harmonics are sinusoidal voltages or currents having frequencies that are integer multiples of the frequency at which the supply system
#   is desgined to operate (termed the fundamental frequency, usually at 50Hz or 60 Hz).

## Implementation note: 
# Generate three sine waves according to the parameters.
# Concatenate the waves together with the noise to produce the harmoincs wave.
# Select a number of values in the wave with the desired time period and
# modify their values to produce the interruptions.
# Can vary the harmonics-related or interruption-related parameters (voltage decrease and time duration)
# to create different samples of interruptions and harmonics.

import numpy as np
import pandas as pd
import random
import csv
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

# Define the function to generate interruptions and harmonics
def generate_signals(n):
    for i in range(n):
        # Define the harmonics parameters: 
        a3_parameter = (random.choice(range(5, 20, 5))) / 100
        a5_parameter = (random.choice(range(5, 20, 5))) / 100
        a7_parameter = (random.choice(range(5, 20, 5))) / 100
        a1_parameter = (1 - (a3_parameter ** 2) - (a5_parameter ** 2) - (a7_parameter ** 2)) ** 0.5
        # Define the interruption parameters:
        interruption_duration_in_cycles = random.choice(range(1, 10, 3))
        interruption_magnitude = (random.choice(range(0, 11, 2))) / 100
        # Modify the output filename: 
        output_filename = "interruptions_harmonics_sample0" + str(i) + ".csv"

        # Generate the output wave: 
        samples = np.arange(0, SIGNAL_DURATION_IN_SEC, 1 / SAMPLES_PER_CYCLE / SIGNAL_DURATION_IN_CYCLES / WAVE_FREQ * 10)
        noise = np.random.default_rng().normal(scale=NOISE_SD, size=samples.size)

        # Produce the harmonics: 
        a1_wave = a1_parameter * np.sin(2 * np.pi * WAVE_FREQ * samples)
        a3_wave = a3_parameter * np.sin(3 * 2 * np.pi * WAVE_FREQ * samples)
        a5_wave = a5_parameter * np.sin(5 * 2 * np.pi * WAVE_FREQ * samples)
        a7_wave = a7_parameter * np.sin(7 * 2 * np.pi * WAVE_FREQ * samples)
        output_wave = a1_wave + a3_wave + a5_wave + a7_wave + noise

        # Produce the interruptions: 
        interruption_duration_in_samples = int(interruption_duration_in_cycles * SAMPLES_PER_CYCLE)
        interruption_starting_index = int(randrange(samples.size - interruption_duration_in_samples))
        for ii in range(interruption_starting_index, interruption_starting_index + interruption_duration_in_samples):
            output_wave[ii] *= interruption_magnitude

        # # Plot the output wave: 
        # x = np.arange(0, len(output_wave))
        # plt.title("PQDs: Interruptions and harmonics")
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
        for k in range(11):
            file_rows[k].append("")
            file_rows[k].append("")
        file_rows[0][1] = "amplitude (in pu)"
        file_rows[0][3] = "parameters"
        file_rows[1][3] = "waveform frequency"
        file_rows[2][3] = "signal duration in ms"
        file_rows[3][3] = "signal sample per cycle"
        file_rows[4][3] = "noise level in sd"
        file_rows[5][3] = "interruption duration in cycles"
        file_rows[6][3] = "interruption magnitude"
        file_rows[7][3] = "a1 parameter"
        file_rows[8][3] = "a3 parameter"
        file_rows[9][3] = "a5 parameter"
        file_rows[10][3] = "a7 parameter"
        file_rows[0][4] = "values"
        file_rows[1][4] = WAVE_FREQ
        file_rows[2][4] = SIGNAL_DURATION_IN_SEC * 1000
        file_rows[3][4] = SAMPLES_PER_CYCLE
        file_rows[4][4] = NOISE_SD
        file_rows[5][4] = interruption_duration_in_cycles
        file_rows[6][4] = interruption_magnitude
        file_rows[7][4] = a1_parameter
        file_rows[8][4] = a3_parameter
        file_rows[9][4] = a5_parameter
        file_rows[10][4] = a7_parameter
        # Edit the CSV file: 
        output_file = open(output_filename, 'w', newline = '')
        csv_writer = csv.writer(output_file)
        csv_writer.writerows(file_rows)
        output_file.close()
