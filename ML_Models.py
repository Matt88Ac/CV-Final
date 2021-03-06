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
        s = 50
        self.s = (s, s)
        winSize = (40, 40)
        blockSize = (20, 20)
        blockStride = (10, 10)
        cellSize = (5, 5)
        nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        if 'SVModel.dat' in os.listdir():
            self.svm = cv2.ml.SVM_load('SVModel.dat')
            print('SVM Model Loaded!')
            return

        self.svm: cv2.ml_SVM = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1e-6))

        print('Training Model...')

        def openDataSet():
            digits_dataset_n = []
            labels = []

            for element in ['Train', 'Test']:
                for i in range(10):
                    PATH = 'Dataset/' + element + f'/{i}'
                    images = os.listdir(PATH)
                    for image in images:
                        n_Path = PATH + '/' + image
                        image = cv2.imread(n_Path, 0)
                        image = cv2.resize(image, self.s, interpolation=cv2.INTER_CUBIC)
                        digits_dataset_n.append(image)
                        labels.append(i)
            labels = np.array(labels)
            digits_dataset_n = np.array(digits_dataset_n)

            return digits_dataset_n, labels

        dds, lbls = openDataSet()
        xtest, ytest = self.__split_and_train(dds, lbls, split)
        print('Done Training!')

        nbins = self.svm.predict(xtest)
        self.score = accuracy_score(nbins[1], ytest)
        self.con_mat = confusion_matrix(nbins[1], ytest)

        self.svm.save('SVModel.dat')

    def predict(self, digit_image: np.ndarray) -> int:
        # plt.imshow(digit_image, cmap='gray')
        # plt.show()
        if type(digit_image) != np.ndarray:
            return 0

        # im = digit_image.copy().astype(np.uint8)
        im = cv2.resize(digit_image, self.s, interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        test = im.astype('float') / 255
        if test.sum() < 100:
            return 0
        im = np.array(self.hog.compute(im))
        im = im.reshape((1, len(im)))
        pred = self.svm.predict(samples=im)
        return pred[1][0][0]

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
    def __init__(self, split=0.7):
        self.s = (30, 30)

        if 'LogModel.dat' in os.listdir():
            self.logreg = cv2.ml.LogisticRegression_load('LogModel.dat')
            print('Logistic Regression Model Loaded!')
            return
        self.logreg: cv2.ml_LogisticRegression = cv2.ml.LogisticRegression_create()
        self.logreg.setTrainMethod(cv2.ml.LogisticRegression_MINI_BATCH)
        self.logreg.setLearningRate(0.0001)
        self.logreg.setRegularization(cv2.ml.LOGISTIC_REGRESSION_REG_L2)
        self.logreg.setMiniBatchSize(1)
        self.logreg.setIterations(1000)

        def openDataSet():
            digits_dataset_n = []
            labels = []

            for element in ['Train', 'Test']:
                for i in range(10):
                    PATH = 'Dataset/' + element + f'/{i}'
                    images = os.listdir(PATH)
                    for image in images:
                        n_Path = PATH + '/' + image
                        image = cv2.imread(n_Path, 0)
                        image = cv2.resize(image, self.s, interpolation=cv2.INTER_CUBIC)
                        digits_dataset_n.append(image)
                        labels.append(i)
            labels = np.array(labels)
            digits_dataset_n = np.array(digits_dataset_n)

            return digits_dataset_n, labels

        X, Y = openDataSet()

        xtest, ytest = self.__split_and_train(X, Y, split)
        ret, y_pred = self.logreg.predict(xtest)
        print(accuracy_score(y_pred, ytest))

        self.logreg.save('LogModel.dat')

    def __split_and_train(self, dds, labels, split) -> tuple:
        n = len(dds)

        x_train = dds[: int(n * split)]
        y_train = labels[: int(n * split)]

        x_test = dds[int(n * split):]
        y_test = labels[int(n * split):]

        x_train = np.array([x0.flatten() for x0 in x_train], dtype=np.float32)
        x_test = np.array([x0.flatten() for x0 in x_test], dtype=np.float32)

        y_train: np.ndarray = y_train.astype(np.float32)

        self.logreg.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

        return x_test, y_test


class SVM2:
    def __init__(self, plot_confusion_mat=False):
        if 'SVModel.pkl' in os.listdir():
            self.model = joblib.load('SVModel.pkl')

        else:
            self.model = SVC(kernel='rbf', probability=True)
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
        joblib.dump(self.model, 'SVModel.pkl')

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

    def predProb(self, digit: np.ndarray):
        if type(digit) != np.ndarray:
            return 0

        if digit.shape != (50, 50):
            new_im = cv2.resize(digit, (50, 50), interpolation=cv2.INTER_CUBIC)
        else:
            new_im = digit.copy()
        new_im = new_im.astype('float').flatten()
        new_im = np.reshape(new_im, (1, len(new_im)))
        pred = self.model.predict_proba(new_im)[0]
        print(pred)
        return pred.argmax()

# SVM2(plot_confusion_mat=True)
