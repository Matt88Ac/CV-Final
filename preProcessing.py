import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd


class preProcessor:

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image.copy()

        self.binIm = self.__runPP()
        self.original_area = self.findArea()
        self.gray_area, self.gray_M = self.__four_point_transform(self.binIm, self.original_area)
        self.original_area, self.original_M = self.__four_point_transform(self.image, self.original_area)

    def __runPP(self) -> np.ndarray:
        my_filter = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)).astype(np.uint8)

        imag = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        imag = cv2.GaussianBlur(imag, (5, 5), 0)
        imag = cv2.fastNlMeansDenoising(imag, None)
        imag = cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        imag = cv2.bitwise_not(imag, imag)

        kernel = my_filter.copy()
        kernel = np.ones((2, 2), np.uint8)

        imag = cv2.dilate(imag, kernel, iterations=1)
        # imag = cv2.erode(imag, kernel, iterations=1)

        return imag

    def __runPP2(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)).astype(np.uint8)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.fastNlMeansDenoising(gray, None)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 5, 2)
        gray = cv2.bitwise_not(gray, gray)
        gray = cv2.dilate(gray, kernel)
        return gray

    def findArea(self) -> np.ndarray:
        contours, _ = cv2.findContours(self.binIm.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largestContours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        epsilon = 0.015 * cv2.arcLength(largestContours[0], True)
        approx = cv2.approxPolyDP(largestContours[0], epsilon, True)

        return approx

    @staticmethod
    def __four_point_transform(image: np.ndarray, pts: np.ndarray):
        imag = image.copy()

        def order_points(points: np.ndarray):
            rect1 = np.zeros((4, 2), dtype=np.float32)

            s = points.sum(axis=1)
            rect1[0] = points[np.argmin(s)]
            rect1[2] = points[np.argmax(s)]

            diff = np.diff(points, axis=1)
            rect1[1] = points[np.argmin(diff)]
            rect1[3] = points[np.argmax(diff)]

            # return the ordered coordinates
            return rect1

        t_pts = pts.reshape(4, 2).copy()
        # obtain a consistent order of the points and unpack them
        # individually
        rect = order_points(t_pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order

        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(imag, M, (maxWidth, maxHeight), borderMode=cv2.BORDER_TRANSPARENT)
        return warped, M

    def plot(self):
        plt.xticks([])
        plt.yticks([])
        plt.title('Pre Processed Sudoku Grid')
        plt.imshow(self.gray_area, cmap='gray')

        plt.show()


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
