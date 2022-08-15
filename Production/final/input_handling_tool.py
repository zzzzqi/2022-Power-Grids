"""
Name: input_handling_tool.py
Date: 23 Aug 2022
By: Group F Toumetis - MSc CS Final Project - University of Bristol
"""

import os
import re
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
from tensorflow.keras import models, preprocessing
from enum import Enum
import csv
import click

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## ==========================================
# Define the six waveforms as enums
class Waveforms(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'

# Define the column names for the output CSV file
output_csv_columns = [
    "event_id", "start_time", "asset_name", "input_event_csv_filename",
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

# Define the helper function for creating a directory
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Define the helper function for identifying the normal max value of the signal waveform
# First it detects the transformations of the waveform from positive to negative, and vice versa, twice.
# Then it identifies the max value in this specific interval in the waveform.
def identify_max_value(signal):
    count = 0  # The number of the signal change its sign
    start_point = 0  # The first index when the signal change its sign
    end_point = 0  # The second index when the signal change its sign
    index = 0
    max_value = 0
    signal_sign = list(map(lambda x: 1 if x >= 0 else -1, signal * 1000))
    temporary_value = signal_sign[index]

    while index < len(signal) and count < 2:
        if signal_sign[index] != temporary_value:
            temporary_value = signal_sign[index]
            count += 1
            if count == 1:
                start_point = index
            if count == 2:
                end_point = index
                max_value = np.amax(
                    np.fabs(signal.iloc[start_point:end_point]))
                # return the maximum of the array after absoluting the array
                if max_value * 1000 == 0:
                    count -= 1
        index += 1
    return max_value

# Define the helper function for converting all the six signal waveforms into 2D PSR images
def phase_space_graph(import_csv, export_path, tau=20):
    # Load data from the csv file, where index_col is the selected column for the row labels
    signal = pd.read_csv(import_csv, index_col=3)

    # Initialise the image plots
    plt.style.use("grayscale")
    fig, ax = plt.subplots()
    plt.rc("font", size=7)
    plt.xlim(-2, 2, 1)
    plt.ylim(-2, 2, 1)
    plt.axis("on")
    fig.set_size_inches(2, 2)
    image, = ax.plot(0, 0)

    for waveform in Waveforms:
        image_name = export_path + "_{0}.png".format(waveform.value)
        # Normalise the voltage and current data values
        waveform_values = signal[waveform.value].copy()
        max_value = identify_max_value(waveform_values)
        for i in range(len(waveform_values)):
            waveform_values.iloc[i] /= max_value

        image.set_data(waveform_values, np.roll(waveform_values, tau))
        plt.draw()
        fig.savefig(image_name, dpi=100)
        plt.close(fig)

        rgb_image = PIL.Image.open(image_name)
        grayscale_image = rgb_image.convert("L")
        grayscale_image.save(image_name)

## ==========================================
# Define the function for converting input signals into the 2D PSR images
# This function serves as a command line tool option
def convert_signals(input_event_dir, psr_dir, output_csv_filepath):
    # Create the destination directory for storing the images
    mkdir(psr_dir)
    csv_suffix_length = len(".csv")

    # Create the output CSV file for logging the metadata of the input signals
    f = open(output_csv_filepath, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(output_csv_columns)

    # Iterate through the input event files, and convert its six signals to 2D PSR images
    input_event_files = os.listdir(input_event_dir)
    for input_event_file in input_event_files:
        if re.search(".csv$", input_event_file) is not None:
            export_image_name = input_event_file[
                :len(input_event_file) - csv_suffix_length]
            export_path = psr_dir + os.sep + export_image_name
            input_event_path = input_event_dir + os.sep + input_event_file
            phase_space_graph(input_event_path, export_path)

            input_event_path = input_event_dir + os.sep + input_event_file
            input_event_signals = pd.read_csv(input_event_path, index_col=3)
            input_event_row = []
            for i in range(3):
                input_event_row.append(input_event_signals.iloc[0, i])
            input_event_row.append(input_event_file[
                :len(input_event_file) - csv_suffix_length])

            csv_writer.writerow(input_event_row)
    f.close()

# Define the function for using the CNN model to make predictions on the 2D PSR images
# This function serves as a command line tool option
def make_predictions(cnn_model_path, psr_dir, output_csv_filepath):
    print("here")
    # Load the trained CNN model and the 2D PSR images
    cnn = models.load_model(cnn_model_path)
    event_waveform_images = tf.io.gfile.listdir(psr_dir)

    # Iterate through the 2D PSR images, and use the CNN model to make predictions
    # Log the prediction scores in the output CSV file
    pqd_categories = [
        "flickers", "harmonics", "interruptions", "interruptions_harmonics", "osc_transients",
        "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"
    ]
    output_df = pd.read_csv(output_csv_filepath, index_col=3)
    png_suffix_length = len(".png")
    for image_name in event_waveform_images:
        if re.search(".png$", image_name) is not None:
            target_waveform = image_name[
                :len(image_name) - png_suffix_length].split("_")[-1]
            input_event_filename = image_name[
                :len(image_name) - png_suffix_length].rsplit("_", 1)[0]

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
                    target_waveform.lower() + "_" + pqd_categories[i]] \
                     = predictions[0][i]
    # Export the dataframe as a CSV file
    output_df.to_csv(output_csv_filepath)

# Define the function for using the CNN model to make predictions on the input events, 
# with NO images saved locally
# This function serves as a command line tool option
def predict_from_events(cnn_model_path, input_event_dir, output_csv_filepath):
    csv_suffix_length = len(".csv")
    # Load the trained CNN model
    cnn = models.load_model(cnn_model_path)

    # Create the output CSV file with the defined column names
    f = open(output_csv_filepath, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(output_csv_columns)

    # Iterate through the input event files, and convert its six signals to 2D PSR images
    # No images would be saved locally
    input_event_files = os.listdir(input_event_dir)
    for input_event_file in input_event_files:
        if re.search('.csv$', input_event_file) is not None:
            # Read the input event CSV file
            input_event_signals = pd.read_csv(
                input_event_dir + os.sep + input_event_file, 
                index_col=3
            )
            
            # Extract the CSV filename and the event's metadata
            input_event_row = []
            for i in range(3):
                input_event_row.append(input_event_signals.iloc[0, i])
            input_event_row.append(input_event_file[
                :len(input_event_file) - csv_suffix_length])

            # Prepare the image plotting
            plt.style.use('grayscale')
            fig, ax = plt.subplots(dpi=100)
            plt.rc("font", size=7)
            plt.xlim(-2, 2, 1)
            plt.ylim(-2, 2, 1)
            plt.axis('on')
            fig.set_size_inches(2, 2)
            image, = ax.plot(0, 0)

            # Iterate each of the six signal waveforms
            # Normalise the waveform values
            # Create a 2D PSR image for each waveform, but not save it
            # Directly use the CNN model to make predictions on the generated image
            for waveform in Waveforms:
                waveform_values = input_event_signals[waveform.value].copy()
                max_value = identify_max_value(waveform_values)
                for i in range(len(waveform_values)):
                    waveform_values.iloc[i] /= max_value

                image.set_data(waveform_values, np.roll(waveform_values, 20))

                plt.draw()
                io_buf = io.BytesIO()
                fig.savefig(io_buf, format="png", dpi=100)
                io_buf.seek(0)

                im = PIL.Image.open(io_buf)
                prediction_image = im.convert('L')

                prediction_image_array = preprocessing.image.img_to_array(
                    prediction_image
                )
                prediction_image_array = np.array(
                    [prediction_image_array]
                )
                predictions = cnn.predict(prediction_image_array)
                for i in range(10):
                    input_event_row.append(predictions[0][i])

                io_buf.close()
                plt.close(fig)
            # Write the data of the input event to the output CSV file
            csv_writer.writerow(input_event_row)
    f.close()

## ==========================================
## Define the command line tool options
@click.command()
@click.option(
    "--convert",
    default=False,
    help="Convert input events to 2D PSR images.")
@click.option(
    "--predict",
    default=False,
    help="Make predictions on the PQD types of each signal.")
@click.option(
    '--no_images', 'noimages',
    default=False,
    help="Make predictions on input events with NO images saved")
@click.option(
    '--output_name',
    default='cnn_output',
    help="Change the name of the output file.")
@click.argument(
    "filepath",
    type=click.Path(exists=True))

def main(filepath, convert, predict, output_name, noimages):
    """
    This is the command line tool for handling input events. \n
    This tool reads CSV files as inputs, converts the waveforms to 2D PSR images, 
    and uses the trained CNN model to make predictions on their PQD types. \n
    The tool generates one CSV file as its output. \n
    This file contains the event metadata and the predictions of each the signal waveforms. \n
    Use: handleinput [OPTIONS] FILEPATH \n
    Help: handleinput --help
    """
    current_dir = filepath
    output_csv_filepath = current_dir + os.sep + output_name + ".csv"
    input_event_dir = current_dir + os.sep + "event_data"
    psr_dir = current_dir + os.sep + "event_waveform_images"
    cnn_model_path = current_dir + os.sep + "trained_cnn_models" \
         + os.sep + "basic_pqd_cnn.h5"

    if not noimages:
        if convert:
            convert_signals(input_event_dir, psr_dir, output_csv_filepath)
        if predict:
            make_predictions(cnn_model_path, psr_dir, output_csv_filepath)
    else:
        predict_from_events(cnn_model_path, input_event_dir, output_csv_filepath)

if __name__ == '__main__':
    # main('.')
    main()
