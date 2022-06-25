import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras import layers, models, utils, optimizers, losses, metrics


class PG_CNN(object):
    cnn = models.Sequential()
    # Add first convolution layer
    cnn.add(layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        input_shape=(200, 200, 1)
    ))
    cnn.add(layers.AvgPool2D(
        pool_size=(2, 2)
    ))

    # Add second convolution layer
    cnn.add(layers.Conv2D(
        filters=48,
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    ))
    cnn.add(layers.AvgPool2D(
        pool_size=(2, 2)
    ))

    # Add final convolution layer
    cnn.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    ))

    # Add fully-connected layer
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(units=10, activation='softmax'))

    def __int__(self):

        training_path = './pqd_dataset_04/training_set'
        validation_path = './pqd_dataset_04/validation_set'

        self.train_dataset = utils.image_dataset_from_directory(
            directory=training_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            image_size=(200, 200)
        )

        self.validation_dataset = utils.image_dataset_from_directory(
            directory=validation_path,
            labels='inferred',
            label_mode='categorical',
            color_mode='grayscale',
            image_size=(200, 200)
        )

        type_names = self.train_dataset.class_names

        self.df_columns = []

        for each_item in type_names:
            self.df_columns.append(each_item)

        print(self.df_columns)

    def compile(self):
        PG_CNN.cnn.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return None

    def fit(self):
        PG_CNN.cnn.fit(self.train_dataset, validation_data=self.validation_dataset, epochs=10)
        PG_CNN.cnn.save_weights("./ckpt/new_cnn.h5")

    # show probabilities of 10 classes
    def predict(self):
        model_path = "./ckpt/new_cnn.h5"
        PG_CNN.cnn.load_weights(model_path)

        prediction_set = tf.io.gfile.listdir("./pqd_dataset_04/prediction_set")
        print(prediction_set)

        self.event_dic = {}
        cnt = 0

        for each_item in prediction_set:
            prediction_image = utils.load_img(
                path="./pqd_dataset_04/prediction_set" + "/" + each_item,
                color_mode='grayscale',
                target_size=(200, 200)
            )
            prediction_image = utils.img_to_array(prediction_image)
            img = prediction_image.reshape(
                [1, prediction_image.shape[0], prediction_image.shape[1], prediction_image.shape[2]])

            predictions_res = PG_CNN.cnn.predict(img)

            event_index = "event" + str(cnt) + each_item
            self.event_dic[event_index] = predictions_res[0]

            cnt += 1

        self.df_res = pd.DataFrame(self.event_dic, index=self.df_columns).T
        self.df_res.to_csv("./prediction_results/predictions_10_classes.csv")

        print("------ The probabilities of multiple classes ------")
        print(self.df_res)

    # show the most possible result with probability
    def show_most_possible_result(self):
        model_path = "./ckpt/new_cnn.h5"
        PG_CNN.cnn.load_weights(model_path)

        prediction_set = tf.io.gfile.listdir("./pqd_dataset_04/prediction_set")

        cnt = 0
        colum_names = ["result_output", "probability"]
        content = {}

        for each_item in prediction_set:
            prediction_image = utils.load_img(
                path="./pqd_dataset_04/prediction_set" + "/" + each_item,
                color_mode='grayscale',
                target_size=(200, 200)
            )
            prediction_image = utils.img_to_array(prediction_image)
            img = prediction_image.reshape(
                [1, prediction_image.shape[0], prediction_image.shape[1], prediction_image.shape[2]])

            predictions_res = PG_CNN.cnn.predict(img)

            event_index = "event" + str(cnt) + each_item

            index = np.argmax(predictions_res)
            each_event = [self.df_columns[index], predictions_res[0][index]]
            content[event_index] = each_event

            cnt += 1

        df = pd.DataFrame(content, index=colum_names).T
        df.to_csv("./prediction_results/predictions_most_possible_results.csv")

        print("------ The most possible result with probability ------")
        print(df)


if __name__ == '__main__':
    cnn = PG_CNN()
    cnn.__int__()
    # cnn.compile()
    # cnn.fit()
    # cnn.predict()
    cnn.show_most_possible_result()
