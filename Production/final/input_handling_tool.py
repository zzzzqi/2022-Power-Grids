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


# Define the six waveforms as enums
class Waveforms(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'


# Define helper function for making directories
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# The function first detect the positive and negative transformation of the signal twice,
# then using the value between these two times as a signal interval for detecting extreme values
# Function for detect the value of the first extreme point in the signal
def find_extreme_value(signal):
    count = 0  # The number of the signal change its sign
    start_point = 0  # The first index when the signal change its sign
    end_point = 0  # The second index when the signal change its sign
    index = 0
    max_value = 0
    signal_copy = signal * 1000
    signal_sign = np.sign(signal_copy.astype(int))  # np.sign: return -1 if x<0; 0 if x==0; 1 if x>0
    temporary_value = signal_sign.iloc[index]  # The sign of the first point of the signal

    while index < len(signal) and count < 2:

        if signal_sign.iloc[index] != temporary_value:
            temporary_value = signal_sign.iloc[index]
            count += 1
            if count == 1:
                start_point = index
            if count == 2:
                end_point = index

                max_value = np.amax(np.fabs(signal.iloc[start_point:end_point]))
                # return the maximum of the array after absoluting the array
                if max_value * 1000 == 0:
                    count -= 1
        index += 1

    return max_value


# Define helper function for converting the six waveforms at the same time
def phase_space_graph(import_csv, export_path, tau=20):
    # Load data from the csv file
    path = 'event_data' + os.sep + import_csv
    signal = pd.read_csv(path, index_col=3)  # index_col means choose which col as the row labels

    plt.style.use('grayscale')
    fig, ax = plt.subplots()
    plt.rc("font", size=7)
    plt.xlim(-2, 2, 1)
    plt.ylim(-2, 2, 1)
    plt.axis('on')
    fig.set_size_inches(2, 2)

    image, = ax.plot(0, 0)  # initialize plot

    for waveform in Waveforms:
        image_name = export_path + "_{0}.png".format(waveform.value)

        # Normalise the voltage and current data values to between -1 and 1
        waveform_values = signal[waveform.value].copy()
        max_value = find_extreme_value(waveform_values)
        # print(image_name + ":{0:5f}".format(max_value))  # print the max value
        for i in range(len(waveform_values)):
            waveform_values.iloc[i] /= max_value

        image.set_data(waveform_values, np.roll(waveform_values, tau))
        plt.draw()
        fig.savefig(image_name, dpi=100)
        plt.close(fig)

        rgb_image = PIL.Image.open(image_name)
        grayscale_image = rgb_image.convert("L")
        grayscale_image.save(image_name)


# Define the column names for the output csv file
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


def convert_signals(input_event_dir, psr_dir, output_csv_filepath):
    ## Step 1: Generate 2D images from the event signals
    mkdir(psr_dir)
    csv_suffix_length = len(".csv")

    # Create the output csv file with the column names
    f = open(output_csv_filepath, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(output_csv_columns)

    # Convert the event waveforms into the 2D PSR images
    files = os.listdir(input_event_dir)
    for filename in files:
        if re.search(".csv$", filename) is not None:
            export_image_name = filename[:len(filename) - csv_suffix_length]  # delete the suffix of csv file
            export_path = psr_dir + os.sep + export_image_name
            phase_space_graph(filename, export_path)

            input_event_path = input_event_dir + os.sep + filename
            input_event_signals = pd.read_csv(input_event_path, index_col=4)
            input_event_row = []
            for i in range(3):
                input_event_row.append(input_event_signals.iloc[0, i])
            input_event_row.append(filename[:len(filename) - csv_suffix_length])

            csv_writer.writerow(input_event_row)
    f.close()


def make_predictions(cnn_model_path, psr_dir, output_csv_filepath):
    ## Step 2: Use the CNN model to classify the images

    # Load the trained CNN model and the generated images
    cnn = models.load_model(cnn_model_path)
    event_waveform_images = tf.io.gfile.listdir(psr_dir)

    # Iterate through the generated images, and use the CNN model to make predictions
    # Log the prediction scores in the output CSV file
    pqd_categories = [
        "flickers", "harmonics", "interruptions", "interruptions_harmonics", "osc_transients",
        "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"
    ]
    output_df = pd.read_csv(output_csv_filepath, index_col=3)
    png_suffix_length = len(".png")
    for image_name in event_waveform_images:
        if re.search(".png$", image_name) is not None:
            target_waveform = image_name[:len(image_name) - png_suffix_length].split("_")[-1]
            input_event_filename = image_name[:len(image_name) - png_suffix_length].rsplit("_", 1)[0]

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
    output_df.to_csv(output_csv_filepath)


# The function called by command tool
def prediction_from_signal(cnn_model_path, input_event_dir, output_csv_filepath):
    # Load the trained CNN model
    cnn = models.load_model(cnn_model_path)

    # Create the output csv file with the column names
    f = open(output_csv_filepath, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(output_csv_columns)  # load the header of the table

    # Conversion of events to 2D images
    files = os.listdir(input_event_dir)
    for file in files:
        if re.search('.csv$', file) is not None:

            signal = pd.read_csv(input_event_dir + os.sep + file,
                                 index_col=3)  # index_col means choose which col as the row labels

            event_list = []
            event_list.append(file[:len(file) - len(".csv")])
            for i in range(3):
                event_list.append(signal.iloc[0, i])

            plt.style.use('grayscale')
            fig, ax = plt.subplots(dpi=100)
            plt.rc("font", size=7)

            plt.xlim(-2, 2, 1)
            plt.ylim(-2, 2, 1)
            plt.axis('on')
            fig.set_size_inches(2, 2)

            image, = ax.plot(0, 0)  # initialize plot

            for waveform in Waveforms:

                # Normalise the voltage and current data values to between -1 and 1
                waveform_values = signal[waveform.value].copy()
                max_value = find_extreme_value(waveform_values)
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
                    event_list.append(predictions[0][i])

                io_buf.close()
                plt.close(fig)

            csv_writer.writerow(event_list)

    f.close()


## Define the command line tool options
@click.command()
@click.option(
    "--convert",
    default=True,
    help="Convert the signals to PSR images.")
@click.option(
    "--predict",
    default=True,
    help="Make predictions on the PQD types of each signal.")
@click.option(
    '--output_name',
    default='cnn_output',
    help="Change the name of the output file.")
@click.option(
    '--no_images', 'noimages',
    default=False,
    help="This option will prediect the signal without exporting images")
@click.argument(
    "filepath",
    type=click.Path(exists=True))
def main(filepath, convert, predict, output_name, noimages):
    """
    This is the command line tool for handling input events. \n
    This tool reads CSV files as inputs, converts the waveforms to PSR images, 
    and uses the trained CNN model to generate predictions on their PQD types. \n
    The tool generates one CSV file as the output. \n
    Use: handleinput [OPTIONS] FILEPATH \n
    Help: handleinput --help
    """
    current_dir = filepath
    output_csv_filepath = current_dir + os.sep + output_name + ".csv"
    input_event_dir = current_dir + os.sep + "event_data"
    psr_dir = current_dir + os.sep + "event_waveform_images"
    cnn_model_path = current_dir + os.sep + "trained_cnn_models" + os.sep + "basic_pqd_cnn.h5"

    if noimages:
        prediction_from_signal(cnn_model_path, input_event_dir, output_csv_filepath)

    else:
        if convert:
            convert_signals(input_event_dir, psr_dir, output_csv_filepath)

        if predict:
            make_predictions(cnn_model_path, psr_dir, output_csv_filepath)


if __name__ == '__main__':
    # main('.')
    main()
