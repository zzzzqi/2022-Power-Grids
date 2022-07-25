import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from keras import layers, models, utils, optimizers, losses, metrics


class LAYERSPLUS_CNN(object):
    cnn = models.Sequential()
    # Add first convolution layer
    cnn.add(layers.Conv2D(
        filters=32,
        kernel_size=(7, 7),
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
        kernel_size=(5, 5),
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
    cnn.add(layers.AvgPool2D(
        pool_size=(2, 2)
    ))

    cnn.add(layers.Conv2D(
        filters=80,
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    ))
    cnn.add(layers.AvgPool2D(
        pool_size=(2, 2)
    ))

    # Add fully-connected layer
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(units=10, activation='softmax'))

    def __int__(self, training_path='./pqd_dataset_05/training_set', validation_path='./pqd_dataset_05/validation_set'):

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

    def compile(self):
        LAYERSPLUS_CNN.cnn.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return None

    def fit(self, model_name):
        saved_path = "./ckpt/" + model_name + ".h5"
        LAYERSPLUS_CNN.cnn.fit(self.train_dataset, validation_data=self.validation_dataset, epochs=10)
        LAYERSPLUS_CNN.cnn.save_weights(saved_path)

    # show probabilities of 10 classes
    def predict(self, model_name):
        model_path = "./ckpt/" + model_name + ".h5"
        LAYERSPLUS_CNN.cnn.load_weights(model_path)

        prediction_set = tf.io.gfile.listdir("./pqd_dataset_05/prediction_set")

        self.event_dic = {}
        cnt = 0

        for each_item in prediction_set:
            prediction_image = utils.load_img(
                path="./pqd_dataset_05/prediction_set" + "/" + each_item,
                color_mode='grayscale',
                target_size=(200, 200)
            )
            prediction_image = utils.img_to_array(prediction_image)
            img = prediction_image.reshape(
                [1, prediction_image.shape[0], prediction_image.shape[1], prediction_image.shape[2]])

            predictions_res = LAYERSPLUS_CNN.cnn.predict(img)

            event_index = "event_" + str(cnt) + "_" + each_item
            self.event_dic[event_index] = predictions_res[0]

            cnt += 1

        self.df_res = pd.DataFrame(self.event_dic, index=self.df_columns).T
        results_path = "./prediction_results/" + model_name + "_predictions_10_classes.csv"
        self.df_res.to_csv(results_path)

        print("------ The probabilities of multiple classes ------")
        print(self.df_res)

    # show the most possible result with probability
    def show_most_possible_result(self, model_path="../trained_cnn_model/basic_pqd_cnn.h5"):

        LAYERSPLUS_CNN.cnn.load_weights(model_path)

        prediction_set = tf.io.gfile.listdir("./pqd_dataset_05/prediction_set")

        cnt = 0
        colum_names = ["result_output", "probability"]
        content = {}

        for each_item in prediction_set:
            prediction_image = utils.load_img(
                path="./pqd_dataset_05/prediction_set" + "/" + each_item,
                color_mode='grayscale',
                target_size=(200, 200)
            )
            prediction_image = utils.img_to_array(prediction_image)
            img = prediction_image.reshape(
                [1, prediction_image.shape[0], prediction_image.shape[1], prediction_image.shape[2]])

            predictions_res = LAYERSPLUS_CNN.cnn.predict(img)

            event_index = "event_" + str(cnt) + "_" + each_item

            index = np.argmax(predictions_res)
            each_event = [self.df_columns[index], predictions_res[0][index]]
            content[event_index] = each_event

            cnt += 1

        df = pd.DataFrame(content, index=colum_names).T
        df.to_csv("./prediction_results/predictions_most_possible_results.csv")

        print("------ The most possible result with probability ------")
        print(df)

    def evaluation(self, model_name):
        # 初始化变量
        tp = 0
        fn = 0
        fp = 0
        tn = 0

        # 初始化cm
        content = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

        ]

        cm = pd.DataFrame(data=content, index=self.df_columns, columns=self.df_columns)
        print(cm)
        print()

        folder_name_list = self.train_dataset.class_names
        print("evaluation in process……")
        real_list = []
        pred_list = []
        name_list = []

        # folder_name must match the actual_label
        for each_folder in folder_name_list:
            each_images_folder = "./pqd_dataset_05/validation_set" + "/" + str(each_folder)
            each_type_images_set = tf.io.gfile.listdir(each_images_folder)

            each_tp = 0
            each_fn = 0

            for each_img in each_type_images_set:
                real = each_folder

                if each_img.endswith(".png"):
                    prediction_image = utils.load_img(
                        path=each_images_folder + "/" + each_img,
                        color_mode='grayscale',
                        target_size=(200, 200)
                    )
                    prediction_image = utils.img_to_array(prediction_image)
                    img = prediction_image.reshape(
                        [1, prediction_image.shape[0], prediction_image.shape[1], prediction_image.shape[2]])

                    # .h5 file load in predict method
                    predictions_res = LAYERSPLUS_CNN.cnn.predict(img)
                    index = np.argmax(predictions_res)
                    predict = self.df_columns[index]

                    real = real.strip()
                    each_folder = each_folder.strip()
                    predict = predict.strip()

                    real_list.append(real)
                    pred_list.append(predict)
                    name_list.append(each_folder)

                    if real == each_folder and predict == each_folder:
                        tp = tp + 1
                        each_tp = each_tp + 1
                        cm[predict][real] = each_tp

                    elif real == each_folder and predict != each_folder:
                        fn = fn + 1
                        each_fn = each_fn + 1
                        cm[predict][real] = each_fn

                    elif real != each_folder and predict == each_folder:
                        fp = fp + 1

                    elif real != each_folder and predict != each_folder:
                        tn = tn + 1

                else:
                    print("PNG format images are required.")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1_score = 2 * ((precision * recall) / (precision + recall))

        print("tp: %s" % tp)
        print("fn: %s" % fn)
        print("fp: %s" % fp)
        print("tn: %s" % tn)

        # print("name list: %s" % name_list)
        # print("pred list: %s" % pred_list)
        # print("real list: %s" % real_list)
        # print(len(name_list))
        # print(len(pred_list))
        # print(len(real_list))

        print("------ confusion_matrix is stated below ------")
        print(cm)
        matrix_name = model_name + "_confusion_matrix" + ".csv"
        cm.to_csv("./confusion_matrix/" + matrix_name)

        print()

        print("------ evaluation results are stated below ------")
        print("precision: %.6f" % precision)
        print("recall: %.6f" % recall)
        print("F1_score: %.6f" % F1_score)

        return None


if __name__ == '__main__':

    cnn = LAYERSPLUS_CNN()

    model_name = "layersplus_cnn"
    cnn.__int__()
    cnn.compile()
    cnn.fit(model_name)
    cnn.predict(model_name)
    cnn.evaluation(model_name)
