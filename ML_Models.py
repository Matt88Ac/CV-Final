from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC as SVC
import numpy as np
import cv2
import joblib
import os

class DigitsSVM:

    def __init__(self, split: float = 0.7):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1e-6))

        s = 50
        self.s = (s, s)

        def openDataSet():
            digits_dataset = cv2.imread('digitst.jpg', 0)
            digits_dataset: np.ndarray = np.array([np.hsplit(row, 9) for
                                                   row in np.vsplit(digits_dataset, 40)]).reshape(-1, 2500)

            digits_dataset: np.ndarray = digits_dataset.reshape((360, 50, 50))
            digits_dataset_n = np.zeros((360, s, s), dtype=np.uint8)
            labels = np.tile(np.arange(1, 10), int(len(digits_dataset) / 9))

            for i in range(360):
                digits_dataset_n[i] = cv2.resize(digits_dataset[i], self.s, interpolation=cv2.INTER_CUBIC)

            return digits_dataset_n, labels

        winSize = (40, 40)
        blockSize = (20, 20)
        blockStride = (10, 10)
        cellSize = (5, 5)
        nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

        dds, lbls = openDataSet()
        xtest, ytest = self.__split_and_train(dds, lbls, split)

        nbins = self.svm.predict(xtest)
        self.score = accuracy_score(nbins[1], ytest)
        self.con_mat = confusion_matrix(nbins[1], ytest)

    def predict(self, digit_image: np.ndarray) -> int:
        if type(digit_image) != np.ndarray:
            return 0

        im = cv2.resize(digit_image, dsize=self.s, interpolation=cv2.INTER_CUBIC)
        im: np.ndarray = im.reshape(self.s).astype(np.uint8)
        test = im.astype('float') / 255
        if test.sum() < 100:
            return 0
        im = self.hog.compute(im)
        return self.svm.predict(np.array([im]))[1][0][0]

    def __split_and_train(self, dds, labels, split) -> tuple:
        n = len(dds)

        x_train = dds[: int(n * split)]
        y_train = labels[: int(n * split)]

        x_test = dds[int(n * split):]
        y_test = labels[int(n * split):]

        x_train = np.array([self.hog.compute(x0) for x0 in x_train], dtype=np.float32)
        x_test = np.array([self.hog.compute(x0) for x0 in x_test], dtype=np.float32)

        self.svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train.astype(np.int32))

        return x_test, y_test


class LogisticRegDigits:
    def __init__(self):
        self.logreg = cv2.ml.LogisticRegression_create()


class SVM2:
    def __init__(self):
        if 'SVModel.pkl' in os.listdir():
            self.model = joblib.load('SVModel.pkl')
        else:





