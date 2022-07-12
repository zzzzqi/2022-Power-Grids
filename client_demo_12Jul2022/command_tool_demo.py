from email.policy import default
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models, preprocessing
from enum import Enum
import csv
import click

# df_columns_predictions = ["event_id", "start_time", "asset_name", "manual_event_type", "input_event_csv_filename", 
#                 "vab_flickers", "vab_harmonics", "vab_interruptions", "vab_interruptions_harmonics", "vab_osc_transients",
#                 "vab_sags", "vab_sags_harmonics", "vab_spikes", "vab_swells", "vab_swells_harmonics", "vbc_flickers", 
#                 "vbc_harmonics", "vbc_interruptions", "vbc_interruptions_harmonics", "vbc_osc_transients", "vbc_sags", 
#                 "vbc_sags_harmonics", "vbc_spikes", "vbc_swells", "vbc_swells_harmonics", "vca_flickers", 
#                 "vca_harmonics", "vca_interruptions", "vca_interruptions_harmonics", "vca_osc_transients", 
#                 "vca_sags", "vca_sags_harmonics", "vca_spikes", "vca_swells", "vca_swells_harmonics", "ia_flickers", 
#                 "ia_harmonics", "ia_interruptions", "ia_interruptions_harmonics", "ia_osc_transients", "ia_sags", 
#                 "ia_sags_harmonics", "ia_spikes", "ia_swells", "ia_swells_harmonics", "ib_flickers", "ib_harmonics", 
#                 "ib_interruptions", "ib_interruptions_harmonics", "ib_osc_transients", "ib_sags", "ib_sags_harmonics", 
#                 "ib_spikes", "ib_swells", "ib_swells_harmonics", "ic_flickers", "ic_harmonics", "ic_interruptions", 
#                 "ic_interruptions_harmonics", "ic_osc_transients", "ic_sags", "ic_sags_harmonics", "ic_spikes", 
#                 "ic_swells", "ic_swells_harmonics"]

# # Six waveforms
# class Waveforms(Enum):
#     Vab = 'Vab'
#     Vbc = 'Vbc'
#     Vca = 'Vca'
#     Ia = 'Ia'
#     Ib = 'Ib'
#     Ic = 'Ic'   

# def mkdir(path):
#     folder = os.path.exists(path)
#     if not folder:
#         os.makedirs(path)

# # Function for converting six waveforms at the same time
# def phase_space_graph(import_csv, export_path, tau=20):
#     # Load data from the csv file
#     path = 'event_data' + os.sep + import_csv
#     signal = pd.read_csv(path, index_col=4)  # index_col means choose which col as the row labels

#     plt.style.use('grayscale')
#     fig, ax = plt.subplots()
#     plt.rc("font", size=7)
#     plt.xlim(-2, 2, 1)
#     plt.ylim(-2, 2, 1)
#     plt.axis('on')
#     fig.set_size_inches(2, 2)
    
#     image, = ax.plot(0,0) # initialize plot
    
#     for waveform in Waveforms:
#         image.set_data(signal[waveform.value], np.roll(signal[waveform.value], tau))
#         plt.draw()
#         fig.savefig(export_path+"_{0}.png".format(waveform.value), dpi=100)
#         colour_to_gray(export_path+"_{0}.png".format(waveform.value))
#         plt.close(fig)


# # Function for convert the colour space from RGB to gray scale
# def colour_to_gray(image_name):
#     colour_image = Image.open(image_name)
#     gray_image = colour_image.convert('L')
#     gray_image.save(image_name)




# @click.command()
# @click.option('--convert/--no-convert', default=True,
#                 help='convert the signal to image')
# @click.option('--predict/--no-predict', default=True, 
#                 help='predict the disturbance type')
# # @click.option('--tau', default=20, 
# #                 help='')
# @click.option('--outname', default='output',
#                 help='the name of the output file')
# @click.argument('filepath', type=click.Path(exists=True), 
#                 )

# def main(filepath, convert, predict, outname):
#     """The command tool for part one."""

#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     current_dir = filepath # get current folder


#     # PSR directory
#     input_event_dir = current_dir + os.sep + 'event_data'
#     psr_dir = current_dir + os.sep + 'event_waveform_images'
#     mkdir(psr_dir)
#     csv_suffix = '.csv$'

    
#     if convert:

#         f = open(outname+'.csv', 'w')
#         # create the csv writer
#         writer = csv.writer(f)
#         # write a row to the csv file, here it is the attributes of the table
#         writer.writerow(df_columns_predictions)

#         # Conversion of events to 2D images
#         files = os.listdir(input_event_dir)
#         for file in files:
#             if re.search(csv_suffix, file) is not None:
#                 export_image = file[:len(file)-4] # delete the suffix of csv file
#                 export_path = psr_dir + os.sep + export_image
#                 phase_space_graph(file, export_path)
#                 path = input_event_dir+os.sep+file
#                 signal = pd.read_csv(path,index_col=4)
#                 event_list = []
#                 for i in range(3):
#                     event_list.append(signal.iloc[0,i])
#                 event_list.append('test_label')
#                 event_list.append(file[:len(file)-4])
#                 writer.writerow(event_list)
#         f.close()



#     if predict:

#         ### Feed images into CNN model for scores ###

#         # Load the trained CNN model
#         model_name = "pqd_cnn_test01_dataset05_model.h5"
#         # model_path = current_directory + os.sep + model_name
#         model_path = current_dir + os.sep + "trained_cnn_model" + os.sep + model_name
#         cnn = models.load_model(model_path)

#         # Import the prediction dataset
#         prediction_set_path = current_dir + os.sep + "event_waveform_images"
#         prediction_set = tf.io.gfile.listdir(prediction_set_path)

#         # Iterate through the dataset, and make predictions with the trained CNN model.
#         # Log the prediction scores in the pd dataframe

#         col_names = ["flickers", "harmonics", "interruptions", "interruptions_harmonics",
#                         "osc_transients", "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"]
        
#         output_file = pd.read_csv(outname+'.csv', index_col=4)
        
#         for image_name in prediction_set:
#             if re.search(".png$", image_name) is not None:
#                 wave = image_name[:len(image_name)-4].split("_")[-1]
#                 input_event_csv_filename = image_name[:len(image_name)-4].rsplit("_", 1)[0]

#                 prediction_image = preprocessing.image.load_img(
#                     path=prediction_set_path + "/" + image_name,
#                     color_mode='grayscale',
#                     target_size=(200, 200)
#                 )
#                 prediction_image_array = preprocessing.image.img_to_array(
#                     prediction_image
#                 )
#                 prediction_image_array = np.array(
#                     [prediction_image_array]
#                 )
#                 predictions = cnn.predict(prediction_image_array)
#                 for i in range(10):
#                     output_file.loc[input_event_csv_filename, wave.lower()+'_'+col_names[i]] = predictions[0][i]            ### Use input_event_csv_filname as the index to identify which cell to go into

#         # change the positions of columns
#         # output_file.reset_index(inplace=True) # option 'inplace' means keep the change
#         # output_file = output_file.reindex(columns=df_columns_predictions)


#         # Save the pd dataframe as a CSV file
#         output_file.to_csv(current_dir + os.sep + outname +".csv")

# if __name__ == '__main__':
#     main()



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
# def phase_space_graph(import_csv, export_path, tau=20):
#     # Read the target input event file from the "event_data" directory
#     event_file = "event_data" + os.sep + import_csv
#     event_signals = pd.read_csv(event_file, index_col=4)  # index_col means choose which col as the row labels

#     for waveform in Waveforms:
#         image_name = export_path + "_{0}.png".format(waveform.value)

#         plt.style.use('grayscale')
#         fig, ax = plt.subplots()
#         plt.rc("font", size=7)
#         plt.xlim(-2, 2, 1)
#         plt.ylim(-2, 2, 1)
#         plt.axis('on')
#         fig.set_size_inches(2, 2)
#         image, = ax.plot(0, 0) # initialize plot

#         image.set_data(event_signals[waveform.value], np.roll(event_signals[waveform.value], tau))
#         plt.draw()
#         fig.savefig(image_name, dpi=100)
#         fig.clear()
#         plt.close(fig)
#         plt.close("all")

#         rgb_image = Image.open(image_name)
#         grayscale_image = rgb_image.convert("L")
#         grayscale_image.save(image_name)

def phase_space_graph(import_csv, export_path, tau=20):
    # Load data from the csv file
    path = 'event_data' + os.sep + import_csv
    signal = pd.read_csv(path, index_col=4)  # index_col means choose which col as the row labels

    plt.style.use('grayscale')
    fig, ax = plt.subplots()
    plt.rc("font", size=7)
    plt.xlim(-2, 2, 1)
    plt.ylim(-2, 2, 1)
    plt.axis('on')
    fig.set_size_inches(2, 2)
    
    image, = ax.plot(0,0) # initialize plot
    
    for waveform in Waveforms:
        image_name = export_path + "_{0}.png".format(waveform.value)

        image.set_data(signal[waveform.value], np.roll(signal[waveform.value], tau))
        plt.draw()
        fig.savefig(image_name, dpi=100)
        plt.close(fig)

        rgb_image = Image.open(image_name)
        grayscale_image = rgb_image.convert("L")
        grayscale_image.save(image_name)

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

def convert_process(psr_dir, output_csv_filename, input_event_dir, csv_suffix):
    mkdir(psr_dir)

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

def predict_process(current_dir, psr_dir, output_csv_filename):
    ## Step 2: Use the CNN model to classify the images
    # Load the trained CNN model and the generated images
    # cnn_model_name = "pqd_cnn_test01_dataset05_model.h5"
    cnn_model_name = 'basic_pqd_cnn.h5'
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




@click.command()
@click.option('--convert', default=True,
                help='convert the signal to image')
@click.option('--predict', default=True, 
                help='predict the disturbance type')
# @click.option('--tau', default=20, 
#                 help='')
@click.option('--outname', default='output',
                help='the name of the output file')
@click.argument('filepath', type=click.Path(exists=True), 
                )

def main(filepath, convert, predict, outname):
    """The command tool for part one."""
        # Define the input and output filenames and directories
    # current_dir = os.getcwd()
    # output_csv_filename = "cnn_output.csv"
    current_dir = filepath
    output_csv_filename = outname+".csv"

    input_event_dir = current_dir + os.sep + "event_data"
    psr_dir = current_dir + os.sep + "event_waveform_images"
    csv_suffix = ".csv$"

    if convert:
        convert_process(psr_dir, output_csv_filename, input_event_dir, csv_suffix)
    
    if predict:
        predict_process(current_dir, psr_dir, output_csv_filename)
    
if __name__ == '__main__':
    main()