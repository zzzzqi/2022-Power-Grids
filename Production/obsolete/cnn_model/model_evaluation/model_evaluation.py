import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, preprocessing

trained_models_dir = os.getcwd() + os.sep + "saved_models"
trained_models = tf.io.gfile.listdir(trained_models_dir)
target_dataset_dir = os.getcwd() + os.sep + "target_dataset"
target_dataset = tf.io.gfile.listdir(target_dataset_dir)

pqd_categories = [
        "flickers", "harmonics", "interruptions", "interruptions_harmonics", "osc_transients",
        "sags", "sags_harmonics", "spikes", "swells", "swells_harmonics"
    ]

for model in trained_models:
    if re.search(".h5$", model) is not None:
        target_model = models.load_model(trained_models_dir + os.sep + model)
        target_inputs_dict = {category: 0 for category in pqd_categories}
        target_outputs_dict = {category: {output_cat: 0 for output_cat in pqd_categories} for category in pqd_categories}

        image_count = 0
        for image in target_dataset:
            if re.search(".png$", image) is not None:
                print(image_count)
                image_count += 1

                target_image = preprocessing.image.load_img(
                    path=target_dataset_dir + os.sep + image,
                    color_mode='grayscale',
                    target_size=(200, 200)
                )
                target_image_array = preprocessing.image.img_to_array(
                    target_image
                )
                target_image_array = np.array(
                    [target_image_array]
                )
                target_predictions = target_model.predict(target_image_array)
                target_score = tf.nn.softmax(target_predictions[0])
                target_output = pqd_categories[np.argmax(target_score)]

                if ("sags_harmonics" in image):
                    target_inputs_dict["sags_harmonics"] += 1
                    target_outputs_dict["sags_harmonics"][target_output] += 1
                elif ("swells_harmonics" in image):
                    target_inputs_dict["swells_harmonics"] += 1
                    target_outputs_dict["swells_harmonics"][target_output] += 1
                elif ("interruptions_harmonics" in image):
                    target_inputs_dict["interruptions_harmonics"] += 1
                    target_outputs_dict["interruptions_harmonics"][target_output] += 1
                elif ("sags" in image):
                    target_inputs_dict["sags"] += 1
                    target_outputs_dict["sags"][target_output] += 1
                elif ("swells" in image):
                    target_inputs_dict["swells"] += 1
                    target_outputs_dict["swells"][target_output] += 1
                elif ("interruptions" in image):
                    target_inputs_dict["interruptions"] += 1
                    target_outputs_dict["interruptions"][target_output] += 1
                elif ("flickers" in image):
                    target_inputs_dict["flickers"] += 1
                    target_outputs_dict["flickers"][target_output] += 1
                elif ("spikes" in image):
                    target_inputs_dict["spikes"] += 1
                    target_outputs_dict["spikes"][target_output] += 1
                elif ("osc_transients" in image):
                    target_inputs_dict["osc_transients"] += 1
                    target_outputs_dict["osc_transients"][target_output] += 1
                elif ("harmonics" in image):
                    target_inputs_dict["harmonics"] += 1
                    target_outputs_dict["harmonics"][target_output] += 1

        target_df_columns = ["category"]
        for category in pqd_categories:
            target_df_columns.append(category)
        target_df = pd.DataFrame([target_df_columns])
        for key in target_outputs_dict.keys():
            key_row = [key]
            for category in pqd_categories:
                key_row.append(target_outputs_dict[key][category])
            target_df = pd.concat([target_df, pd.DataFrame([key_row])], ignore_index=True)
        target_df = pd.concat([target_df, pd.DataFrame([""])], ignore_index=True)

        target_classification_accuracy = 0
        target_correct_count = 0
        for key in target_outputs_dict.keys():
            target_correct_count += target_outputs_dict[key][key]
        target_prediction_count = 0
        for key in target_outputs_dict.keys():
            target_prediction_count += sum(target_outputs_dict[key].values())
        target_classification_accuracy = target_correct_count / target_prediction_count
        target_df = pd.concat([target_df, 
            pd.DataFrame([["Classification accuracy", "%.5f" % target_classification_accuracy]])], ignore_index=True)
        target_df = pd.concat([target_df, pd.DataFrame([""])], ignore_index=True)

        target_f1_score = 0
        for key in target_outputs_dict.keys():
            key_true_positives = target_outputs_dict[key][key]
            key_false_positives = 0
            for alt_key in target_outputs_dict.keys():
                if alt_key == key: continue
                key_false_positives += target_outputs_dict[alt_key][key]
            key_false_negatives = sum(target_outputs_dict[key].values()) - target_outputs_dict[key][key]

            if key_true_positives == 0:
                key_f1_score = 0
            else:
                key_precision = key_true_positives/ (key_true_positives + key_false_positives)
                key_recall = key_true_positives/ (key_true_positives + key_false_negatives)
                key_f1_score = 2 * (key_precision * key_recall)/ (key_precision + key_recall)
            target_f1_score += key_f1_score
        target_f1_score /= len(pqd_categories)
        target_df = pd.concat([target_df, 
            pd.DataFrame([["Macro-F1 score", "%.5f" % target_f1_score]])], ignore_index=True)

        target_df.to_csv(os.getcwd() + os.sep + "model_evaluation" + os.sep + model[:-3] + "_evaluation.csv")
