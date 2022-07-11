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

    def __int__(self, training_path='./pqd_dataset_05/training_set', validation_path='./pqd_dataset_05/validation_set'):

        # training_path = './pqd_dataset_05/training_set'
        # validation_path = './pqd_dataset_05/validation_set'

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
    def predict(self, model_path="../trained_cnn_model/basic_pqd_cnn.h5"):
        # model_path = "./ckpt/new_cnn.h5"
        PG_CNN.cnn.load_weights(model_path)

        prediction_set = tf.io.gfile.listdir("./pqd_dataset_05/prediction_set")
        # print("94. prediction set 是 %s" % prediction_set)

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

            predictions_res = PG_CNN.cnn.predict(img)

            event_index = "event_" + str(cnt) + "_" + each_item
            self.event_dic[event_index] = predictions_res[0]

            cnt += 1

        self.df_res = pd.DataFrame(self.event_dic, index=self.df_columns).T
        self.df_res.to_csv("./prediction_results/predictions_10_classes.csv")

        print("------ The probabilities of multiple classes ------")
        print(self.df_res)

    # show the most possible result with probability
    def show_most_possible_result(self, model_path="../trained_cnn_model/basic_pqd_cnn.h5"):
        # model_path = "../ckpt/new_cnn.h5"
        PG_CNN.cnn.load_weights(model_path)

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

    def evaluation(self):
        # 初始化变量
        tp = 0
        fn = 0
        fp = 0
        tn = 0

        # 用test_set数据，validation
        # validation_folder = tf.io.gfile.listdir("./pqd_dataset_05/validation_set")
        folder_name_list = self.train_dataset.class_names
        # print(folder_name_list)
        print("evaluation in process……")
        real_list = []
        pred_list = []
        name_list = []

        # folder_name must match the actual_label
        for each_folder in folder_name_list:
            each_images_folder = "./pqd_dataset_05/validation_set" + "/" + str(each_folder)
            # print("each_images_folder是: " + each_images_folder)
            each_type_images_set = tf.io.gfile.listdir(each_images_folder)

            for each_img in each_type_images_set:
                real = each_folder
                # print("real标、签是：" + real)
                # print("each_img是 " + each_img)
                # print(each_images_folder + "/" + each_img)

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
                    predictions_res = PG_CNN.cnn.predict(img)
                    index = np.argmax(predictions_res)

                    predict = self.df_columns[index]
                    # print("202. predict 是：%s" % predict)

                    # 真发生，预测发生
                    # print("real: %s" % type(real))
                    # print("foldername: %s" % type(folder_name))
                    # print("predict: %s" % type(predict))
                    real = real.strip()
                    each_folder = each_folder.strip()
                    predict = predict.strip()

                    real_list.append(real)
                    pred_list.append(predict)
                    name_list.append(each_folder)

                    if real == each_folder and predict == each_folder:
                        tp = tp + 1
                    # 真发生，预测没发生
                    elif real == each_folder and predict != each_folder:
                        fn = fn + 1
                    # 真没发生，预测发生
                    elif real != each_folder and predict == each_folder:
                        fp = fp + 1
                    # 真没发生，预测没发生
                    elif real != each_folder and predict != each_folder:
                        tn = tn + 1
                else:
                    print("PNG format image required.")

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1_score = 2 * ((precision * recall) / (precision + recall))

        # print("tp: %s" % tp)
        # print("fn: %s" % fn)
        # print("fp: %s" % fp)
        # print("tn: %s" % tn)

        # print("name list: %s" % name_list)
        # print("pred list: %s" % pred_list)
        # print("real list: %s" % real_list)
        # print(len(name_list))
        # print(len(pred_list))
        # print(len(real_list))

        # Confusion_Matrix visualization
        content = [
            [tp, fn],
            [fp, tn]
        ]
        index_name = ["actual_ture", "actual_false"]
        colum_name = ["predict_positive", "predict_negative"]

        cm = pd.DataFrame(data=content, index=index_name, columns=colum_name)
        print("------ confusion_matrix is stated below ------")
        print(cm)

        print()

        print("------ evaluation results are stated below ------")
        print("precision: %.2f" % precision)
        print("recall: %.2f" % recall)
        print("F1_score: %.2f" % F1_score)

        return None


if __name__ == '__main__':
    cnn = PG_CNN()
    """
    default dataset05, change dataset path for training a new model.
    """
    cnn.__int__()

    """
    being used for training new cnn model
    """
    # cnn.compile()

    """
    1."the new .h5 file will storage in "./ckpt/new_cnn.h5"
    2.output results will storage in "./prediction_results"
    """
    # cnn.fit()

    """
    the default .h5 file use ../trained_cnn_model/basic_pqd_cnn.h5
    """
    cnn.predict()

    """
    change .h5 in predict method, if we would like to use the new trained .h5 file
    """
    # cnn.predict("./ckpt/new_cnn.h5")
    # cnn.show_most_possible_result()

    """ouput confusion matrix and precision, recall score, f1 score"""
    cnn.evaluation()
