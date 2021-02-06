from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn
import numpy as np
import cv2
import joblib
import os
from matplotlib import pyplot as plt


class DigitsSVM:

    def __init__(self, split: float = 0.7):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)

        self.digits_dataset = cv2.imread('digitst.jpg', 0)
        self.digits_dataset: np.ndarray = np.array([np.hsplit(row, 9) for
                                                    row in np.vsplit(self.digits_dataset, 40)]).reshape(-1, 2500)

        self.digits_dataset: np.ndarray = self.digits_dataset.reshape((360, 50, 50))

        winSize = (40, 40)
        blockSize = (20, 20)
        blockStride = (10, 10)
        cellSize = (5, 5)
        nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        self.labels = np.tile(np.arange(1, 10), int(len(self.digits_dataset) / 9))
        xtest, ytest = self.__split_and_train(split)

        nbins = self.svm.predict(xtest)
        self.score = accuracy_score(nbins[1], ytest)
        self.con_mat = confusion_matrix(nbins[1], ytest)

    def predict(self, digit_image: np.ndarray) -> int:
        if type(digit_image) != np.ndarray:
            return 0

        im = cv2.resize(digit_image, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
        im = im.reshape(50, 50).astype(np.uint8)
        if im.sum() < 1000:
            return 0
        im = self.hog.compute(im)
        return self.svm.predict(np.array([im]))[1][0][0]

    def __split_and_train(self, split) -> tuple:
        n = len(self.digits_dataset)

        x_train = self.digits_dataset[: int(n * split)]
        y_train = self.labels[: int(n * split)]

        x_test = self.digits_dataset[int(n * split):]
        y_test = self.labels[int(n * split):]

        x_train = np.array([self.hog.compute(x0) for x0 in x_train], dtype=np.float32)
        x_test = np.array([self.hog.compute(x0) for x0 in x_test], dtype=np.float32)

        self.svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32))

        return x_test, y_test


class RandomForestDigits:
    def __init__(self, plot_confusion_mat=False):
        if 'RFModel.pkl' in os.listdir():
            self.model = joblib.load('RFModel.pkl')

        else:
            self.model = RandomForestClassifier()
            train_x = []
            train_y = []
            test_x = []
            test_y = []

            for element in ['Train', 'Test']:
                for i in range(10):
                    PATH = 'Dataset/' + element + f'/{i}'
                    images = os.listdir(PATH)
                    for image in images:
                        n_Path = PATH + '/' + image
                        image = cv2.imread(n_Path, 0)
                        image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_CUBIC)
                        if element == 'Train':
                            image = image.flatten()  # .astype('float') / 255
                        else:
                            image = image.flatten()  # .astype('float') / 255
                        eval(element.lower() + '_x.append(image)')
                        eval(element.lower() + f'_y.append({i})')

            test_x = np.array(test_x)
            test_y = np.array(test_y)
            train_x = np.array(train_x)
            train_y = np.array(train_y)

            self.model.fit(train_x, train_y)
            if plot_confusion_mat:
                plt.title('Score: {}'.format(accuracy_score(self.model.predict(test_x), test_y)))
                plot_confusion_matrix(self.model, test_x, test_y, ax=plt.gca())
                plt.show()
        joblib.dump(self.model, 'RFModel.pkl')

    def predict(self, digit: np.ndarray):
        if type(digit) != np.ndarray:
            return 0

        if digit.shape != (50, 50):
            new_im = cv2.resize(digit, (50, 50), interpolation=cv2.INTER_CUBIC)
        else:
            new_im = digit.copy()
        new_im = new_im.astype('float').flatten()
        new_im = np.reshape(new_im, (1, len(new_im)))
        pred = self.model.predict(new_im)
        return pred
