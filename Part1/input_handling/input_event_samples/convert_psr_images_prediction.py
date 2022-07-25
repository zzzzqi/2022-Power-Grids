import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum


# Six waveforms
class Waveforms(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'


# Function for generating table which consist of x, y which y delay tau of x
def generate_2d_phase_space(table, tau):
    ps = table[['amplitude (in pu)']].copy()
    ps.rename(columns={'amplitude (in pu)': 'x'}, inplace=True)
    ps['y'] = np.roll(table['amplitude (in pu)'], tau)
    return ps


# Function for converting six waveforms at the same time
def phase_space_graph(import_csv, export_path, tau=20):
    # Load data from the csv file
    signal = pd.read_csv(import_csv, index_col=4)  # index_col means choose which col as the row labels

    plt.style.use('grayscale')
    fig, ax = plt.subplots()
    plt.rc("font", size=7)
    plt.xlim(-2, 2, 1)
    plt.ylim(-2, 2, 1)
    plt.axis('on')
    fig.set_size_inches(2, 2)
    
    image, = ax.plot(0,0) # initialize plot
    
    for waveform in Waveforms:
        image.set_data(signal[waveform.value], np.roll(signal[waveform.value], tau))
        plt.draw()
        fig.savefig(export_path+"_{0}.png".format(waveform.value), dpi=100)
        colour_to_gray(export_path+"_{0}.png".format(waveform.value))
        # plt.close("all") # With this statement, the program will take more time


# Function for convert the colour space from RGB to gray scale
def colour_to_gray(image_name):
    colour_image = Image.open(image_name)
    gray_image = colour_image.convert('L')
    gray_image.save(image_name)


# Set the regex patterns
csv_suffix = '.csv$'
png_suffix = '.png'

current_directory = os.getcwd() # Identify the current directory and the psr folder directory
files = os.listdir(current_directory)


# All the images are stored in current directory
for import_file in files:
    if re.search(csv_suffix, import_file) is not None:
        export_image = import_file[:len(import_file)-4] # delete the suffix of csv file
        phase_space_graph(import_file, export_image)

# Each event has its own folder
# for import_file in files:
#     if re.search(csv_suffix, import_file) is not None:
#         file_folder_path = current_directory + os.sep + import_file[:len(import_file)-4]
#         if not os.path.exists(file_folder_path):
#             os.makedirs(file_folder_path)
        
#         export_image = file_folder_path + os.sep + import_file[:len(import_file)-4]
#         phase_space_graph(import_file, export_image)
