import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import utils, models, preprocessing
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_dir = os.getcwd()  # get current folder

### turn waveforms into 2D images ###

# Six waveforms
class Waveforms(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'

plt.rc("font", size=7)

# check if the phaseSpace folder exist
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('create successful')


# Function for converting six waveforms at the same time
def phase_space_graph(import_csv, export_path, tau=20):
    # Load data from the csv file
    path = 'input_event_samples' + os.sep + import_csv
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

# PSR directory

input_event_dir = current_dir + os.sep + 'input_event_samples'
psr_dir = current_dir + os.sep + 'prediction_data'
mkdir(psr_dir)
csv_suffix = '.csv$'
png_suffix = '.png'

# Conversion of events to 2D images
# files = os.listdir(input_event_dir)
# for file in files:
#     if re.search(csv_suffix, file) is not None:
#         export_image = file[:len(file)-4] # delete the suffix of csv file
#         export_path = psr_dir + os.sep + export_image
#         phase_space_graph(file, export_path)

#TODO: feed into CNN for scores

### Feed images into CNN model for scores ###

# Load the trained CNN model
model_name = "pqd_cnn_test01_dataset05_model.h5"
# model_path = current_directory + os.sep + model_name
model_path = current_dir + os.sep + "trained_models" + os.sep + model_name
cnn = models.load_model(model_path)

# Define a pd dataframe
df_columns = ["image_name", "flickers", "harmonics", "interruptions", "interruptions_harmonics",
                "osc_transients", "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"]
df = pd.DataFrame(columns=df_columns)

# Import the prediction dataset
prediction_set_path = current_dir + os.sep + "prediction_data"
prediction_set = tf.io.gfile.listdir(prediction_set_path)

# Iterate through the dataset, and make predictions with the trained CNN model.
# Log the prediction scores in the pd dataframe
count = 0
for image_name in prediction_set:
    if re.search(".png$", image_name) is not None:
        prediction_image = preprocessing.image.load_img(
            path=prediction_set_path + "/" + image_name,
            color_mode='grayscale',
            target_size=(200, 200)
        )
        prediction_image_array = preprocessing.image.img_to_array(
            prediction_image
        )
        prediction_image_array = np.array(
            [prediction_image_array]
        )
        predictions = cnn.predict(prediction_image_array)
        df_row = [image_name]
        for prediction in predictions[0]:
            df_row.append(prediction)
        df.loc[count] = df_row
        count += 1

# Save the pd dataframe as a CSV file
df.to_csv(current_dir + os.sep + "predictions.csv") 