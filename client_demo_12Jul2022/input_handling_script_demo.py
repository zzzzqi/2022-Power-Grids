import os
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, preprocessing
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## Step 1: Generate 2D images from the event waveforms
# Define the six waveforms as enums
class Waveforms(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'

# Define function for making directories
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define function for converting the six waveforms at the same time
def phase_space_graph(import_csv, export_path, tau=20):
    # Read the target input event file from the "event_data" directory
    event_file = "event_data" + os.sep + import_csv
    event_signals = pd.read_csv(event_file, index_col=4)  # index_col means choose which col as the row labels

    for waveform in Waveforms:
        image_name = export_path + "_{0}.png".format(waveform.value)

        plt.style.use('grayscale')
        fig, ax = plt.subplots()
        plt.rc("font", size=7)
        plt.xlim(-2, 2, 1)
        plt.ylim(-2, 2, 1)
        plt.axis('on')
        fig.set_size_inches(2, 2)
        image, = ax.plot(0, 0) # initialize plot

        image.set_data(event_signals[waveform.value], np.roll(event_signals[waveform.value], tau))
        plt.draw()
        fig.savefig(image_name, dpi=100)
        fig.clear()
        plt.close(fig)
        plt.close("all")

        rgb_image = Image.open(image_name)
        grayscale_image = rgb_image.convert("L")
        grayscale_image.save(image_name)

# Define the input and output filenames and directories
current_dir = os.getcwd()
input_event_dir = current_dir + os.sep + "event_data"
psr_dir = current_dir + os.sep + "event_waveform_images"
mkdir(psr_dir)
output_csv_filename = "cnn_output.csv"
csv_suffix = ".csv$"

# Define the column names for the output csv file
output_csv_columns = [
    "event_id", "start_time", "asset_name", "manual_event_type", "input_event_csv_filename", 
    "vab_flickers", "vab_harmonics", "vab_interruptions", "vab_interruptions_harmonics", "vab_osc_transients",
    "vab_sags", "vab_sags_harmonics", "vab_spikes", "vab_swells", "vab_swells_harmonics",
    "vbc_flickers", "vbc_harmonics", "vbc_interruptions", "vbc_interruptions_harmonics", "vbc_osc_transients",
    "vbc_sags", "vbc_sags_harmonics", "vbc_spikes", "vbc_swells", "vbc_swells_harmonics", 
    "vca_flickers", "vca_harmonics", "vca_interruptions", "vca_interruptions_harmonics", "vca_osc_transients", 
    "vca_sags", "vca_sags_harmonics", "vca_spikes", "vca_swells", "vca_swells_harmonics", 
    "ia_flickers", "ia_harmonics", "ia_interruptions", "ia_interruptions_harmonics", "ia_osc_transients", 
    "ia_sags", "ia_sags_harmonics", "ia_spikes", "ia_swells", "ia_swells_harmonics", 
    "ib_flickers", "ib_harmonics", "ib_interruptions", "ib_interruptions_harmonics", "ib_osc_transients", 
    "ib_sags", "ib_sags_harmonics", "ib_spikes", "ib_swells", "ib_swells_harmonics", 
    "ic_flickers", "ic_harmonics", "ic_interruptions", "ic_interruptions_harmonics", "ic_osc_transients", 
    "ic_sags", "ic_sags_harmonics", "ic_spikes", "ic_swells", "ic_swells_harmonics"
    ]

# Create the output csv file with the column names
f = open(output_csv_filename, 'w')
csv_writer = csv.writer(f)
csv_writer.writerow(output_csv_columns)

# Convert the event waveforms into the 2D PSR images
files = os.listdir(input_event_dir)
for filename in files:
    if re.search(csv_suffix, filename) is not None:
        export_image_name = filename[:len(filename) - len(".csv")] # delete the suffix of csv file
        export_path = psr_dir + os.sep + export_image_name
        phase_space_graph(filename, export_path)

        input_event_path = input_event_dir + os.sep + filename
        input_event_signals = pd.read_csv(input_event_path, index_col=4)
        input_event_row = [] 
        for i in range(3):
            input_event_row.append(input_event_signals.iloc[0, i])
        input_event_row.append('test_label')
        input_event_row.append(filename[:len(filename) - len(".csv")])

        csv_writer.writerow(input_event_row)
f.close()

## Step 2: Use the CNN model to classify the images
# Load the trained CNN model and the generated images
cnn_model_name = "pqd_cnn_test01_dataset05_model.h5"
cnn_model_path = current_dir + os.sep + "trained_cnn_model" + os.sep + cnn_model_name
cnn = models.load_model(cnn_model_path)
event_waveform_images = tf.io.gfile.listdir(psr_dir)

# Iterate through the generated images, and use the CNN model to make predictions
# Log the prediction scores in the output CSV file
pqd_categories = [
    "flickers", "harmonics", "interruptions", "interruptions_harmonics", "osc_transients",
     "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"
    ]
output_df = pd.read_csv(output_csv_filename, index_col=4)
for image_name in event_waveform_images:
    if re.search(".png$", image_name) is not None:
        target_waveform = image_name[:len(image_name)-4].split("_")[-1]
        input_event_filename = image_name[:len(image_name)-4].rsplit("_", 1)[0]

        target_image = preprocessing.image.load_img(
            path=psr_dir + os.sep + image_name,
            color_mode='grayscale',
            target_size=(200, 200)
        )
        target_image_array = preprocessing.image.img_to_array(
            target_image
        )
        target_image_array = np.array(
            [target_image_array]
        )
        predictions = cnn.predict(target_image_array)
        for i in range(10):
            output_df.loc[
                input_event_filename, 
                target_waveform.lower() + "_" + pqd_categories[i]] = predictions[0][i]

# Save the dataframe as a CSV file
output_df.to_csv(current_dir + os.sep + output_csv_filename)
