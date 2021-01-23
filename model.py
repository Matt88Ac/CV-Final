import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os


class DigitsClassifierNetwork:
    def __init__(self):
        self.model = self.create_model()

    @staticmethod
    def create_model():
        if 'yakov' in os.listdir():
            model = keras.models.load_model("yakov")
            return model
        num_classes = 10
        input_shape = (28, 28, 1)

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

        batch_size = 128
        epochs = 15

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        score = model.evaluate(x_test, y_test, verbose=0)

        model.save("yakov")
        print("Accuracy:", score[1])
        return model

    def predict(self, image: np.ndarray):
        if type(image) != np.ndarray:
            return 0
        if len(image.shape) == 3:
            new_image = cv2.cvtColor(cv2.resize(image.astype(np.uint8), (28, 28), interpolation=cv2.INTER_CUBIC),
                                     cv2.COLOR_BGR2GRAY)
        else:
            new_image = cv2.resize(image.astype(np.uint8), (28, 28), interpolation=cv2.INTER_CUBIC)
        return self.model.predict(new_image.reshape(1, 28, 28, 1).astype('float32'), batch_size=1).argmax()



