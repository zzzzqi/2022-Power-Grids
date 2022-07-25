import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum

# An iterable object about disturbance type
class Disturbance(Enum):
    """The collection of disturbance"""

    fl = 'flickers'
    ha = 'harmonics'
    inter = 'interruptions'
    sags = 'sags'
    swells = 'swells'
    spikes = 'spikes'
    sags_ha = 'sags_harmonics'
    swells_ha = 'swells_harmonics'
    inter_ha = 'interruptions_harmonics'
    osc_transients = 'osc_transients'

# Six waveforms
class Waveform(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'

# Function for making the folder to store the created images
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        os.makedirs(path + os.sep + "prediction_set")
        os.makedirs(path + os.sep + "validation_set")
        os.makedirs(path + os.sep + "training_set")
        os.makedirs(path + os.sep + "testing_set")
        for disturbance_object, disturbance_dir in Disturbance.__members__.items():
            os.makedirs(path + os.sep + "training_set" + os.sep + disturbance_dir)
            os.makedirs(path + os.sep + "validation_set" + os.sep + disturbance_dir)
            os.makedirs(path + os.sep + "testing_set" + os.sep + disturbance_dir)


# Function for generating table which consist of x, y which y delay tau of x
def generate_2d_phase_space(table, tau):
    ps = table[['amplitude (in pu)']].copy()
    ps.rename(columns={'amplitude (in pu)': 'x'}, inplace=True)
    ps['y'] = np.roll(table['amplitude (in pu)'], tau)
    return ps

# Function for plotting the images
def plot_phase_space_graph(import_file, export_file, tau):
    # Load data from the csv file
    signal = pd.read_csv(import_file, index_col=4)  # index_col means choose which col as the row labels
    pf = generate_2d_phase_space(signal, tau)
    # Plot the chart
    fig = plt.figure()
    fig.set_size_inches(2, 2)
    plt.rc("font", size=7)
    plt.style.use('grayscale')
    plt.plot(pf['x'], pf['y'])
    plt.xlim(-2, 2, 1)
    plt.ylim(-2, 2, 1)
    plt.axis('on')
    plt.savefig(export_file, dpi=100)
    # Close the chart for memory management
    fig.clear()
    plt.close(fig)
    plt.close("all")
    colour_to_gray(export_file)

# Function for convert the colour space from RGB to gray scale
def colour_to_gray(image_name):
    colour_image = Image.open(image_name)
    gray_image = colour_image.convert('L')
    gray_image.save(image_name)

# Identify the current directory and the psr folder directory
current_directory = os.getcwd()
psr_folder_directory = current_directory + os.sep + 'psr_images'
# Check if the directory exists. If not, create a new folder
mkdir(psr_folder_directory)
# Set the regex patterns
csv_suffix = '.csv$'
png_suffix = '.png'

# Define the number of samples per PQD type for training, validation and testing
n = 700
# Set the training, validation, and testing set mix
training_mix = 0.6
validation_mix = 0.2
testing_mix = 0.2
# Calculate the corresponding limits for training, validation and testing
training_count = int(n * training_mix)
validation_count = int(n * (training_mix + validation_mix))
flickers_count = 0
osc_count = 0
spikes_count = 0
sags_count = 0
swells_count = 0
interruptions_count = 0
harmonics_count = 0
sags_harmonics_count = 0
swells_harmonics_count = 0
interruptions_harmonics_count = 0

# Process the CSV files and convert them into 2D images
# Store them into the dataset structure: training, validation and testing sets
training_mode = True
files = os.listdir(current_directory)
for file in files:
    if re.search(csv_suffix, file) is not None:
        file_path = current_directory + os.sep + file
        # Flickers
        if (file.find("flickers") != -1):
            if (flickers_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "flickers" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (flickers_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "flickers" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (flickers_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "flickers" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            flickers_count += 1
        # Osc transients
        elif (file.find("osc_transients") != -1):
            if (osc_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "osc_transients" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (osc_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "osc_transients" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (osc_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "osc_transients" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            osc_count += 1
        # Spikes
        elif (file.find("spikes") != -1):
            if (spikes_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "spikes" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (spikes_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "spikes" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (spikes_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "spikes" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            spikes_count += 1
        # Sags and harmonics
        elif (file.find("sags_harmonics") != -1):
            if (sags_harmonics_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "sags_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (sags_harmonics_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "sags_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (sags_harmonics_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "sags_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            sags_harmonics_count += 1
        # Swells and harmonics
        elif (file.find("swells_harmonics") != -1):
            if (swells_harmonics_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "swells_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (swells_harmonics_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "swells_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (swells_harmonics_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "swells_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            swells_harmonics_count += 1
        # Interruptions and harmonics
        elif (file.find("interruptions_harmonics") != -1):
            if (interruptions_harmonics_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "interruptions_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (interruptions_harmonics_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "interruptions_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (interruptions_harmonics_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "interruptions_harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            interruptions_harmonics_count += 1
        # Sags
        elif (file.find("sags") != -1):
            if (sags_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "sags" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (sags_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "sags" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (sags_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "sags" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            sags_count += 1
        # Swells
        elif (file.find("swells") != -1):
            if (swells_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "swells" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (swells_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "swells" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (swells_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "swells" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            swells_count += 1
        # Interruptions
        elif (file.find("interruptions") != -1):
            if (interruptions_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "interruptions" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (interruptions_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "interruptions" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (interruptions_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "interruptions" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            interruptions_count += 1
        # Harmonics
        elif (file.find("harmonics") != -1):
            if (harmonics_count < training_count):
                image_path = psr_folder_directory + os.sep + "training_set" + os.sep + \
                            "harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (harmonics_count < validation_count):
                image_path = psr_folder_directory + os.sep + "validation_set" + os.sep + \
                            "harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            elif (harmonics_count < n):
                image_path = psr_folder_directory + os.sep + "testing_set" + os.sep + \
                            "harmonics" + os.sep + re.sub(csv_suffix, png_suffix, file)
            else:
                image_path = psr_folder_directory + os.sep + "prediction_set" + os.sep + re.sub(csv_suffix, png_suffix, file)
            harmonics_count += 1
        plot_phase_space_graph(file_path, image_path, 20)
        # print("training mode = " + str(training_mode))
        # print("flickers count = " + str(flickers_count))
        # print("harmonics count = " + str(harmonics_count))
        # print("int har count = " + str(interruptions_harmonics_count))
        # print("int count = " + str(interruptions_count))
        # print("osc count = " + str(osc_count))
        # print("sags har count = " + str(sags_harmonics_count))
        # print("sag count = " + str(sags_count))
        # print("spikes count = " + str(spikes_count))
        # print("swells har count = " + str(swells_harmonics_count))
        # print("swells count = " + str(swells_count))
        # print("--------------")
        
