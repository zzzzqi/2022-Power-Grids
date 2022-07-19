import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
from tensorflow.keras import models, preprocessing
from enum import Enum
import csv
import io

output_csv_columns = [
    "input_event_csv_filename", "event_id", "start_time", "asset_name", 
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

current_dir = os.getcwd()  # get current folder

# Six waveforms
class Waveforms(Enum):
    Vab = 'Vab'
    Vbc = 'Vbc'
    Vca = 'Vca'
    Ia = 'Ia'
    Ib = 'Ib'
    Ic = 'Ic'


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

# PSR directory
input_event_dir = current_dir + os.sep + 'event_data'
psr_dir = current_dir + os.sep + 'event_waveform_images'
mkdir(psr_dir)
csv_suffix_length = len(".csv")

# Load the trained CNN model
model_name = "basic_pqd_cnn.h5"
model_path = current_dir + os.sep + "trained_cnn_models" + os.sep + model_name
cnn = models.load_model(model_path)

f = open('output_withoutimages.csv', 'w')
# create the csv writer
writer = csv.writer(f)
# write a row to the csv file, here it is the attributes of the table
writer.writerow(output_csv_columns)

# Conversion of events to 2D images
files = os.listdir(input_event_dir)
for file in files:
    if re.search('.csv$', file) is not None:

        signal = pd.read_csv('event_data' + os.sep + file, index_col=4)  # index_col means choose which col as the row labels

        event_list = []
        event_list.append(file[:len(file) - csv_suffix_length])
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
            image.set_data(signal[waveform.value], np.roll(signal[waveform.value], 20))
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

        writer.writerow(event_list)

f.close()
