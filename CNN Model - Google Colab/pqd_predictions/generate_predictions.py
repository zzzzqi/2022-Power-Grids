# generate_predictions.py

import os
import re
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import models, utils, preprocessing

# Load the trained CNN model
model_name = "model01.h5"
current_directory = os.getcwd()
model_path = current_directory + os.sep + "trained_models" + os.sep + model_name
cnn = models.load_model(model_path)

# Define a pd dataframe
df_columns = ["image_name", "flickers", "harmonics", "interruptions", "interruptions_harmonics",
                "osc_transients", "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"]
df = pd.DataFrame(columns=df_columns)

# Import the prediction dataset
prediction_set_path = current_directory + os.sep + "prediction_data"
prediction_set = tf.io.gfile.listdir(prediction_set_path)

# Iterate through the dataset, and make predictions with the trained CNN model.
# Log the prediction scores in the pd dataframe
count = 0
for image_name in prediction_set:
    if re.search(".png$", image_name) is not None:
        prediction_image = utils.load_img(
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
df.to_csv(current_directory + os.sep + "predictions.csv") 
