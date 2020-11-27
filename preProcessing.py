import numpy as np
import cv2
from matplotlib import pyplot as plt


class preProcessing:

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image.copy()

        self.binIm = self.__runPP()
        self.original_area = self.__findArea()
        self.gray_area, self.gray_M = self.__four_point_transform(self.binIm, self.original_area)
        self.original_area, self.original_M = self.__four_point_transform(self.image, self.original_area)

        # self.gray_area: np.ndarray = self.gray_area[10:, :]
        # self.original_area: np.ndarray = self.original_area[10:, :]

    def __runPP(self) -> np.ndarray:
        imag = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        imag = cv2.GaussianBlur(imag, (9, 9), 0)
        imag = cv2.fastNlMeansDenoising(imag, None)
        imag = cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        imag = cv2.bitwise_not(imag, imag)

        kernel = np.ones((2, 2), np.uint8)

        imag = cv2.dilate(imag, kernel, iterations=1)

        return imag

    def __findArea(self) -> np.ndarray:
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
        plt.subplot(2, 2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title('Original Sudoku Grid')
        plt.imshow(self.original_area)

        plt.subplot(2, 2, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('Pre Processed Sudoku Grid')
        plt.imshow(self.gray_area, cmap='gray')

        plt.show()


class SVM:

    def __init__(self):
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_LINEAR)

        self.digits_dataset = cv2.imread('digitst.jpg', 0)
        self.digits_dataset = np.array([np.hsplit(row, 9) for
                                        row in np.vsplit(self.digits_dataset, 40)]).reshape(-1, 2500)




img = cv2.imread('sudoku.jpg')
pp = preProcessing(img)
pp.plot()
