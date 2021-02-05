import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from preProcessing import preProcessor
from ML_Models import DigitsSVM
import os
from datetime import datetime
import imutils


class Cells:

    def __init__(self, sudoku: np.ndarray):
        self.prep = preProcessor(sudoku)
        color = (50, 120, 200)
        self.raw = np.array_split(self.prep.gray_area, 9)
        self.raw = [np.array_split(row, 9, axis=1) for row in self.raw]
        shapes = np.array([l.shape[1] for r in self.raw for l in r])
        mn = np.min(shapes)
        for i in range(len(self.raw)):
            for j in range(len(self.raw[i])):
                self.raw[i][j] = self.raw[i][j][:, :mn]

        self.raw = np.array(self.raw)
        self.raw = self.raw.reshape((81,) + self.raw.shape[2:])

        self.cells = np.zeros((81, 50, 50))
        self.original = np.array_split(self.prep.original_area, 9)
        self.original = [np.array_split(row, 9, axis=1) for row in self.original]
        shapes = np.array([l.shape[1] for r in self.original for l in r])
        mn = np.min(shapes)
        for i in range(len(self.original)):
            for j in range(len(self.original[i])):
                self.original[i][j] = self.original[i][j][:, :mn]
        self.original = np.array(self.original)
        self.original = self.original.reshape((81, 1) + self.original.shape[2:])

        def CompleteClassifier(self, img: np.ndarray) -> np.ndarray:

            def plotLines(imag: np.ndarray, edges: np.ndarray):
                for x1, y1, x2, y2 in edges[:, 0, :]:
                    cv2.line(imag, (x1, y1), (x2, y2), color, 1)
                return imag

            def plotLines2(imag: np.ndarray, edges: np.ndarray):
                height, width = np.shape(img)
                for x1, y1 in edges[:, 0, :]:
                    if y1 != 0:
                        m = -1 / np.tan(y1)
                        c = x1 / np.sin(y1)
                        cv2.line(imag, (0, int(c)), (width, int(m * width + c)), color, thickness=2)
                    else:
                        cv2.line(imag, (x1, 0), (x1, height), color, thickness=2)
                return imag

            # lines = cv2.HoughLinesP(img, 1, np.pi / 40, 180, maxLineGap=280, minLineLength=60)
            lines = cv2.HoughLines(img, 1, np.pi / 180, 360)
            close = plotLines2(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lines)
            self.lines1 = close.copy()
            # plt.imshow(close)
            # plt.show()

            gray = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 3)
            # blur = cv2.GaussianBlur(gray, (3, 3), 0)
            sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
            self.lines2 = sharpen.copy()
            # plt.imshow(sharpen, cmap='gray')
            # plt.show()

            thresh = cv2.threshold(sharpen, 100, 250, cv2.THRESH_BINARY_INV)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=1)
            self.lines3 = close.copy()

            cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # plt.imshow(close, cmap='gray')
            # plt.show()

            areas = np.array([cv2.contourArea(c) for c in cnts])
            lenW = np.array([cv2.boundingRect(c)[2] for c in cnts])
            xx = []
            yy = []

            tempo = np.zeros((81, 50, 50))
            Wmn = lenW.mean()
            Wstd = lenW.std()

            i = 0
            new_img = self.prep.original_area.copy()
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if h == 0 or w == 0:
                    continue
                if 1 / 1.5 > w / h or w / h > 1.5:  # Optionally 1
                    continue
                if w < 3 or h < 3:
                    continue

                cond = np.quantile(areas, 0.1) <= cv2.contourArea(c) <= np.quantile(areas, 0.95)
                if Wmn - Wstd <= w <= Wmn + Wstd:
                    # print(w, h)
                    new_img = cv2.rectangle(new_img, (x, y), (w + x, h + y), color=(240, 70, 130), thickness=3)
                    temp = self.prep.gray_area[y:y + h, x:x + w]
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(temp, cmap='gray')
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(new_img)
                    # plt.show()
                    temp = cv2.resize(temp, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
                    tempo[i] = temp.copy()
                    xx.append(y)
                    yy.append(x)
                    i += 1

            # taking care of positions, as they are in the original image
            tempo = tempo[np.argsort(xx)]
            tempo = tempo.reshape((9, 9, 50, 50))
            yy = np.array(yy)
            yy = yy[np.argsort(xx)].reshape(9, 9)

            for i in range(9):
                tempo[i] = tempo[i][np.argsort(yy[i])]

            tempo = tempo.reshape(81, 50, 50)

            return tempo

        try:
            self.cells = CompleteClassifier(self, self.prep.gray_area)

        except ValueError:
            self.cells = self.raw.copy()

    def __getitem__(self, item) -> np.ndarray:
        return self.cells[item]

    def __improve(self):
        for i in range(81):
            self.cells[i, :3] = 0

    def __setitem__(self, key, value):
        self.cells[key] = value

    def __len__(self):
        return len(self.cells)

    def plot(self, save: str = None):
        k = 1
        if self.cells.sum() > 0:
            for i in range(self.cells.shape[0]):
                plt.subplot(9, 9, k)
                k += 1
                plt.xticks([]), plt.yticks([])
                plt.imshow(self.cells[i], cmap='gray')

            if save is not None:
                plt.savefig(save)
                plt.clf()
                return
            plt.show()

        else:
            for i in range(self.cells.shape[0]):
                plt.subplot(9, 9, k)
                k += 1
                plt.xticks([]), plt.yticks([])
                plt.imshow(self.raw[i], cmap='gray')

            if save is not None:
                plt.savefig(save)
                plt.clf()
                return
            plt.show()


class Digits:

    def __init__(self, sudoku: np.ndarray):
        self.cells = Cells(sudoku)
        # self.cells.plot()
        # self.cells.prep.plot()

        self.svm = DigitsSVM()

        self.digits = np.array([self.__preprocess_image(i) for i in range(len(self.cells.cells))])

        self.images = self.digits[:, 1]
        self.digits = self.digits[:, 0]
        pred_cells = np.array([self.svm.predict(c) for c in self.cells.cells]).reshape((9, 9))

        self.matrix: np.ndarray = np.array([self.svm.predict(d) for d in self.digits]).reshape(9, 9)

    def __extract_digit(self, which, kernel_size: tuple = (5, 5)):
        # im = self.cells.cells[which][0]
        im = self.cells[which].copy().astype(np.uint8)
        # plt.imshow(im, cmap='gray')
        # plt.show()

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
        contours = sorted(contours, key=cv2.contourArea)[::-1]

        if len(contours) != 0:

            # c = max(contours, key=cv2.contourArea)
            # x, y, w, h = cv2.boundingRect(c)
#
            # original_area = original[y:y + h, x:x + w]
            # n_h, n_w = original_area.shape
#
            # # validate contours is number and not noise
            # if n_h > (o_h / 5) and n_w > (o_w / 5):
            #     res = np.zeros((r_h, r_h))
            #     dh = r_h - h
            #     dw = r_w - w
#
            #     res[int(dh / 2):-int(dh / 2) - dh % 2, int(dw / 2):-int(dw / 2) - dw % 2] = original_area
#
            #     cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
            # find the biggest area
            # c = max(contours, key=cv2.contourArea)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)

                original_area = original[y - 5:min(y + h + 5, o_h), x - 5:min(x + w + 5, o_w)]
                if np.sum(original_area.astype('float') / 255) < 10:
                    continue
                #
                # if 1 / 1.4 > w / h or w / h > 1.4:  # Optionally 1
                #     pass
                # plt.imshow(original_area, cmap='gray')
                # plt.plot()

                res = cv2.rectangle(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), (x, y), (x + w, y + h), (200, 150, 70), 2)

                # plt.subplot(1, 3, 1)
                # plt.imshow(original_area, cmap='gray')
                # plt.subplot(1, 3, 2)
                # plt.imshow(im, cmap='gray')
                # plt.subplot(1, 3, 3)
                # plt.imshow(res)
                # plt.show()
                #
                # original_area = cv2.GaussianBlur(original_area, (3, 3), 0)
                # original_area = cv2.fastNlMeansDenoising(original_area, None)
                # original_area = cv2.morphologyEx(original_area, cv2.MORPH_OPEN, kernel)
                # original_area = cv2.morphologyEx(original_area, cv2.MORPH_CLOSE, kernel)

                return original_area, res

        return None, im

    def __preprocess_image(self, which):
        img = self.cells[which].astype(np.uint8)
        rows = np.shape(img)[0]

        for i in range(rows):
            cv2.floodFill(img, None, (0, i), 0)
            cv2.floodFill(img, None, (i, 0), 0)
            cv2.floodFill(img, None, (rows - 1, i), 0)
            cv2.floodFill(img, None, (i, rows - 1), 0)
            cv2.floodFill(img, None, (1, i), 1)
            cv2.floodFill(img, None, (i, 1), 1)
            cv2.floodFill(img, None, (rows - 2, i), 1)
            cv2.floodFill(img, None, (i, rows - 2), 1)

        rowtop = None
        rowbottom = None
        colleft = None
        colright = None
        thresholdBottom = 50
        thresholdTop = 50
        thresholdLeft = 50
        thresholdRight = 50
        center = rows // 2
        for i in range(center, rows):
            if rowbottom is None:
                temp = img[i]
                if sum(temp) < thresholdBottom or i == rows - 1:
                    rowbottom = i
            if rowtop is None:
                temp = img[rows - i - 1]
                if sum(temp) < thresholdTop or i == rows - 1:
                    rowtop = rows - i - 1
            if colright is None:
                temp = img[:, i]
                if sum(temp) < thresholdRight or i == rows - 1:
                    colright = i
            if colleft is None:
                temp = img[:, rows - i - 1]
                if sum(temp) < thresholdLeft or i == rows - 1:
                    colleft = rows - i - 1

        # Centering the bounding box's contents
        newimg = np.zeros(np.shape(img))
        startatX = (rows + colleft - colright) // 2
        startatY = (rows - rowbottom + rowtop) // 2
        for y in range(startatY, (rows + rowbottom - rowtop) // 2):
            for x in range(startatX, (rows - colleft + colright) // 2):
                newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        newimg = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, kernel)
        newimg = cv2.morphologyEx(newimg, cv2.MORPH_CLOSE, kernel)

        return newimg, newimg

    def plot(self, save: str = None):
        for i in range(len(self.cells)):
            plt.subplot(9, 9, i + 1)
            plt.xticks([]), plt.yticks([])
            plt.imshow(self.images[i], cmap='gray')

        if save is not None:
            plt.savefig(save)
            plt.clf()
            return
        plt.show()


class Sudoku:

    def __init__(self, grid: np.ndarray):
        self.digits = Digits(grid)
        self.solution = self.__solve()
        self.sol_grid = self.__drawSolution()
        self.sol_grid = self.sol_grid[:, 0]
        self.sol_grid = self.__cells_to_grid()

    def __solve(self) -> np.ndarray:

        def findNextCellToFill(grid, i, j):
            for x in range(i, 9):
                for y in range(j, 9):
                    if grid[x][y] == 0:
                        return x, y
            for x in range(0, 9):
                for y in range(0, 9):
                    if grid[x][y] == 0:
                        return x, y

            return -1, -1

        def isValid(grid, i, j, e):
            rowOk = all([e != grid[i][x] for x in range(9)])
            if rowOk:
                columnOk = all([e != grid[x][j] for x in range(9)])
                if columnOk:
                    # finding the top left x,y co-ordinates of the section containing the i,j cell
                    secTopX, secTopY = 3 * (i // 3), 3 * (j // 3)  # floored quotient should be used here.
                    for x in range(secTopX, secTopX + 3):
                        for y in range(secTopY, secTopY + 3):
                            if grid[x][y] == e:
                                return False
                    return True
            return False

        def solveSudoku(grid, i=0, j=0):
            i, j = findNextCellToFill(grid, i, j)
            if i == -1:
                return True
            for e in range(1, 10):
                if isValid(grid, i, j, e):
                    grid[i][j] = e
                    if solveSudoku(grid, i, j):
                        return True
                    # Undo the current cell for backtracking
                    grid[i][j] = 0
            return False

        new_grid = self.digits.matrix.copy()
        solveSudoku(new_grid)
        return new_grid

    def __drawSolution(self) -> np.ndarray:
        if type(self.solution) != np.ndarray:
            return np.zeros_like(self.digits.cells.original)

        to_draw: np.ndarray = self.digits.cells.original.copy()
        mat = self.digits.matrix.flatten()
        sol = self.solution.flatten()

        def drawDigit(cell: np.ndarray, digit: int, color: tuple):
            h, w, _ = cell.shape

            size = w / h
            offsetTop = int(h * 0.75 * size)
            offsetLeft = int(w * 0.25 * size)

            font = cv2.FONT_HERSHEY_SIMPLEX
            return cv2.putText(cell.copy(), str(int(digit)), (offsetLeft, offsetTop), font, size, color, 2, cv2.LINE_AA)

        for i in range(81):
            if self.digits.matrix.flatten()[i] == 0:
                to_draw[i][0] = drawDigit(to_draw[i][0], sol[i], (170, 50, 150))

        return to_draw

    def __cells_to_grid(self):
        img = np.zeros_like(self.digits.cells.prep.original_area)

        h = 0
        w = 0
        cell = self.sol_grid[0]
        for i in range(9):
            for j in range(9):
                index = i * 9 + j
                cell = self.sol_grid[index]

                img[h: h + cell.shape[0], w: w + cell.shape[1]] = cell
                w += cell.shape[1]

            w = 0
            h += cell.shape[0]

        return img

    def plot(self, with_original_res=True):
        fig, ax = plt.subplots(1, 2, figsize=(16, 16))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('Original Sudoku Grid')
        ax[0].imshow(self.digits.cells.prep.image)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('Final Solution for the given Sudoku grid')
        if with_original_res:
            ax[1].imshow(cv2.warpPerspective(self.sol_grid, self.digits.cells.prep.original_M,
                                             (self.digits.cells.prep.image.shape[1],
                                              self.digits.cells.prep.image.shape[0]),
                                             dst=self.digits.cells.prep.image, borderMode=cv2.BORDER_TRANSPARENT,
                                             flags=cv2.WARP_INVERSE_MAP))
        else:
            ax[1].imshow(self.sol_grid)

        plt.show()
        return ax[1]


def createVideo(sudoku: np.ndarray):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case

    curdir = os.getcwd().replace(os.getcwd()[2], '/') + '/Maps'
    if not os.path.exists(curdir):
        os.makedirs(curdir)

    sol = Sudoku(sudoku.copy())
    data = sol.digits.matrix
    color = data > 0
    color = color.astype(str)
    color[color == 'True'] = 'gold'
    color[color == 'False'] = 'white'
    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    ax.set_title('via SVM model')
    ax.table(cellText=sol.digits.matrix.astype(int), loc="center", cellColours=color, cellLoc='center', fontsize=3)
    plt.savefig(curdir + '/f11.jpg')
    plt.clf()
    sol.digits.cells.plot(save=curdir + '/f9.jpg')
    sol.digits.plot(save=curdir + '/f10.jpg')

    f1 = sol.digits.cells.prep.image
    f2 = sol.digits.cells.prep.binIm
    f3 = sol.digits.cells.prep.findArea()
    f3 = cv2.drawContours(f1.copy(), [f3], -1, (255, 255, 0), 3)
    f4 = sol.digits.cells.prep.original_area
    f5 = sol.digits.cells.prep.gray_area
    f6 = sol.digits.cells.lines1
    f7 = sol.digits.cells.lines2
    f8 = sol.digits.cells.lines3
    f11 = sol.sol_grid
    f12 = cv2.warpPerspective(sol.sol_grid.copy(), sol.digits.cells.prep.original_M,
                              (sol.digits.cells.prep.image.shape[1],
                               sol.digits.cells.prep.image.shape[0]),
                              dst=sol.digits.cells.prep.image.copy(), borderMode=cv2.BORDER_TRANSPARENT,
                              flags=cv2.WARP_INVERSE_MAP)

    F = [f1, f2, f3, f4, f5, f6, f7, f8, f11, f12]
    inds = list(range(1, 9))
    inds.extend([12, 13])

    h, w, _ = sudoku.shape
    out = cv2.VideoWriter('sudoku_vid.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (w, h))

    for i, j in enumerate(inds):
        cv2.imwrite('Maps/f{}.jpg'.format(j), F[i])

    F = None
    inds = None

    for i in range(1, 14):
        temp = cv2.imread('Maps/f{}.jpg'.format(i))
        temp = cv2.resize(temp, dsize=(w, h), interpolation=cv2.INTER_CUBIC)
        out.write(temp.copy())
        os.remove('Maps/f{}.jpg'.format(i))

    out.release()


image = cv2.imread('data/sudoku.jpg')
createVideo(image)
