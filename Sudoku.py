import numpy as np
import cv2
from matplotlib import pyplot as plt
from preProcessing import DigitsSVM, preProcessor


class Cells:

    def __init__(self, sudoku: np.ndarray):
        self.prep = preProcessor(sudoku)
        color = (50, 120, 200)

        def plotLines(imag: np.ndarray, edges: np.ndarray):
            for x1, y1, x2, y2 in edges[:, 0, :]:
                cv2.line(imag, (x1, y1), (x2, y2), color, 2)
            return imag

        lines = cv2.HoughLinesP(self.prep.gray_area, 1, np.pi / 40, 180, maxLineGap=250, minLineLength=60)
        close = plotLines(cv2.cvtColor(self.prep.gray_area, cv2.COLOR_GRAY2BGR), lines)

        gray = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 3)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        thresh = cv2.threshold(sharpen, 100, 250, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        self.cells = np.zeros((81, 50, 50))

        ww = np.array([cv2.boundingRect(c)[2] for c in cnts])
        xx = []
        yy = []

        i = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if np.quantile(ww, 0.099) <= w <= np.quantile(ww, 0.911):
                temp = self.prep.gray_area[y:y + h, x:x + w]
                temp = cv2.resize(temp, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
                self.cells[i] = temp.copy()
                xx.append(y)
                yy.append(x)
                i += 1

        # taking care of positions, as they are in the original image
        self.cells = self.cells[np.argsort(xx)]
        self.cells = self.cells.reshape((9, 9, 50, 50))
        yy = np.array(yy)
        yy = yy[np.argsort(xx)].reshape(9, 9)

        for i in range(9):
            self.cells[i] = self.cells[i][np.argsort(yy[i])]

        self.cells = self.cells.reshape(81, 50, 50)

    def __getitem__(self, item):
        return self.cells[item]

    def __improve(self):
        for i in range(81):
            self.cells[i, :3] = 0

    def __setitem__(self, key, value):
        self.cells[key] = value

    def __len__(self):
        return len(self.cells)

    def plot(self):
        k = 1
        for i in range(81):
            plt.subplot(9, 9, k)
            k += 1
            plt.imshow(self.cells[i], cmap='gray')
            plt.xticks([]), plt.yticks([])

        plt.show()


class Digits:

    def __init__(self, sudoku: np.ndarray):
        self.cells = Cells(sudoku)
        self.digits = np.array([self.__extract_digit(i) for i in range(len(self.cells))])
        self.images = self.digits[:, 1]
        self.digits = self.digits[:, 0]

        self.svm = DigitsSVM()

        pred_cells = [self.svm.predict(c) for c in self.cells]

        for i in range(len(pred_cells)):
            if pred_cells[i] != 1 and self.digits[i] is None:
                self.digits[i] = self.cells[i].copy()
                self.images[i] = self.cells[i].copy()

        self.matrix = np.array([self.svm.predict(d) for d in self.digits])

        print(self.matrix.reshape(9, 9))

    def __extract_digit(self, which, kernel_size: tuple = (5, 5)):
        im = self.cells[which].copy().astype(np.uint8)
        r_w = 50
        r_h = 50

        original = im.copy()
        o_h, o_w = im.shape

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        for row in range(o_h):
            if im[row, 0] == 255:
                cv2.floodFill(im, None, (0, row), 0)
            if im[row, o_w - 1] == 255:
                cv2.floodFill(im, None, (o_w - 1, row), 0)

        for col in range(o_w):
            if im[0, col] == 255:
                cv2.floodFill(im, None, (col, 0), 0)
            if im[o_h - 1, col] == 255:
                cv2.floodFill(im, None, (col, o_h - 1), 0)

        im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

        # plt.imshow(im, cmap='gray')
        # plt.show()

        contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # find the biggest area
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            original_area = original[y:y + h, x:x + w]
            n_h, n_w = original_area.shape

            # validate contours is number and not noise
            if n_h > (o_h / 5) and n_w > (o_w / 5):
                res = np.zeros((r_h, r_h))
                dh = r_h - h
                dw = r_w - w

                res[int(dh / 2):-int(dh / 2) - dh % 2, int(dw / 2):-int(dw / 2) - dw % 2] = original_area

                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

                return res, im

        return None, im

    def plot(self):
        for i in range(len(self.cells)):
            plt.subplot(9, 9, i + 1)
            plt.imshow(self.images[i], cmap='gray')
            plt.xticks([]), plt.yticks([])

        plt.show()


image = cv2.imread('sudoku.jpg')
digs = Digits(image)
digs.plot()
