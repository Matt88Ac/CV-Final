import numpy as np
import cv2
from matplotlib import pyplot as plt
from preProcessing import preProcessor
from ML_Models import DigitsSVM as DSVM
import os
import warnings

warnings.filterwarnings("ignore")


class Cells:

    def __init__(self, sudoku: np.ndarray):
        self.prep = preProcessor(sudoku)
        # self.prep.plot()

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

        flag = True

        def CompleteClassifier(self, img: np.ndarray, enhance=2) -> np.ndarray:

            def enhanceGrid1(sudGrid: np.ndarray):
                def plotLines(imag: np.ndarray, edgesX: np.ndarray):
                    for x1, y1, x2, y2 in edgesX[:, 0, :]:
                        cv2.line(imag, (x1, y1), (x2, y2), color, 2)
                    return imag

                edges = cv2.Canny(sudGrid, 30, 90, 3)
                lines = cv2.HoughLinesP(sudGrid, 1, np.pi / 40, 180, maxLineGap=250, minLineLength=60, lines=edges)
                close = plotLines(cv2.cvtColor(sudGrid, cv2.COLOR_GRAY2BGR), lines)
                self.lines1 = close.copy()
                # plt.imshow(close)
                # plt.show()

                gray = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
                blur = cv2.medianBlur(gray, 3)
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
                self.lines2 = sharpen.copy()

                thresh = cv2.threshold(sharpen, 100, 250, cv2.THRESH_BINARY_INV)[1]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=1)
                self.lines3 = close.copy()

                return close

            def enhanceGrid2(sudGrid: np.ndarray):
                def plotLines(imag: np.ndarray, edgesX: np.ndarray):
                    for line in edgesX:
                        rho = line[0][0]
                        theta = line[0][1]
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * a)
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * a)
                        cv2.line(imag, (x1, y1), (x2, y2), color, 3)
                    return imag

                edges = cv2.Canny(cv2.blur(sudGrid, (3, 3)), 30, 90, 3)
                lines = cv2.HoughLines(edges, 2, np.pi / 180, 280)
                close = plotLines(cv2.cvtColor(sudGrid, cv2.COLOR_GRAY2BGR), lines)
                self.lines1 = close.copy()
                # plt.imshow(close)
                # plt.show()

                gray = cv2.cvtColor(close, cv2.COLOR_BGR2GRAY)
                blur = cv2.medianBlur(gray, 3)
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
                self.lines2 = sharpen.copy()

                thresh = cv2.threshold(sharpen, 100, 250, cv2.THRESH_BINARY_INV)[1]
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                close = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, kernel, iterations=1)
                self.lines3 = close.copy()

                return close

            img_temp = self.prep.original_area.copy()
            if enhance == 2:
                enhanced = enhanceGrid2(img)
            elif enhance == 1:
                enhanced = enhanceGrid1(img)

            # plt.imshow(enhanced, cmap='gray')
            # plt.show()

            cnts = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts) == 2:
                cnts, hierarchy = cnts
                hierarchy = hierarchy[0]
            elif len(cnts) == 3:
                _, cnts, hierarchy = cnts
                hierarchy = hierarchy[0]
            else:
                hierarchy = None
            cnts = np.array(cnts)
            ww = np.array([cv2.boundingRect(c)[2] for c in cnts])
            hh = np.array([cv2.boundingRect(c)[3] for c in cnts])

            filt = ww == 0
            filt += hh == 0
            filt = ~filt

            ww = ww[filt]
            hh = hh[filt]
            cnts = cnts[filt]
            hierarchy = hierarchy[filt]

            wh = ww * hh

            def exemineContours(cont):
                for i, c in enumerate(cont):
                    x, y, w, h = cv2.boundingRect(c)
                    cond = 1 / 1.4 <= w / h <= 1.4
                    if not cond:
                        continue

                    cond = ww.mean() - ww.std() <= w <= ww.mean() + ww.std()
                    cond += hh.mean() - hh.std() <= h <= hh.mean() + hh.std()
                    # cond *= wh.mean() - wh.std() <= w*h <= wh.mean() + wh.std()
                    if not cond:
                        continue

                    if type(hierarchy) == np.ndarray:
                        if hierarchy[i][3] != -1:
                            continue

                    cv2.rectangle(img_temp, (x, y), (x + w, y + h), color=(200, 40, 150), thickness=3)
                plt.imshow(img_temp)
                plt.show()

            # exemineContours(cnts)
            xx = []
            yy = []
            areas = []
            tempo = self.cells.copy()
            i = 0
            img_temp = self.prep.original_area.copy()
            for j, c in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(c)
                arr = w * h
                if w < 5 or h < 5:
                    continue
                cond = 1 / 1.4 <= w / h <= 1.4
                if not cond:
                    continue

                cond = ww.mean() - ww.std() <= w <= ww.mean() + ww.std()
                cond += hh.mean() - hh.std() <= h <= hh.mean() + hh.std()
                if not cond:
                    continue

                if type(hierarchy) == np.ndarray:
                    if hierarchy[i][3] != -1:
                        continue
                temp = self.prep.gray_area[y:y + h, x:x + w]
                temp = cv2.resize(temp, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(img_temp, (x, y), (x + w, y + h), color=(200, 40, 150), thickness=3)
                # if j % 5 == 0:
                #    plt.imshow(img_temp)
                # plt.pause(0.01)
                # plt.show()
                tempo[i] = temp.copy()
                xx.append(y)
                yy.append(x)
                # areas.append(arr)
                i += 1
            # plt.imshow(img_temp)
            # plt.show()

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
            print('Cells detected.')

        except ValueError:
            print('Error in cells detector, trying another method...')
            if flag:
                flag = False
                self.cells = CompleteClassifier(self, self.prep.gray_area, 1)
                print('Other method succeeded.')
            else:
                self.cells = self.raw.copy()
                print('Other method failed.')

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
                plt.imshow(self.raw[i][0], cmap='gray')

            if save is not None:
                plt.savefig(save)
                plt.clf()
                return
            plt.show()


class Digits:

    def __init__(self, sudoku: np.ndarray, model: DSVM = None):
        self.cells = Cells(sudoku)
        # self.cells.plot()
        self.digits = np.array([self.__extract_digit(i) for i in range(len(self.cells.raw))])

        self.images = self.digits[:, 1]
        self.digits = self.digits[:, 0]
        if model is None:
            self.svm = DSVM()
        else:
            self.svm = model
        pred_cells = [self.svm.predict(c) for c in self.cells]
        # print(np.array(pred_cells).reshape(9, 9))

        for i in range(len(pred_cells)):
            if self.images[i].sum() < self.cells[i].sum() and pred_cells[i] != 1 and self.digits[i] is None:
                self.digits[i] = self.cells[i].copy()
                self.images[i] = self.cells[i].copy()

        pred_cells = np.array(pred_cells).reshape(9, 9)
        self.matrix = np.zeros((9, 9))
        i = 0
        j = 0
        for d in self.digits:
            pred = self.svm.predict(d)
            if pred == 0:
                j += 1
                i = i + int(j == 9)
                j = j % 9
                continue
            if pred in self.matrix[i, :]:
                self.matrix[i, j] = pred_cells[i, j]
            elif pred in self.matrix[:, j]:
                self.matrix[i, j] = pred_cells[i, j]
            else:
                self.matrix[i, j] = pred
            j += 1
            i = i + int(j == 9)
            j = j % 9

        print(self.matrix)

    def __extract_digit(self, which, kernel_size: tuple = (5, 5)):
        im = self.cells.raw[which]  # self.cells[which].copy().astype(np.uint8)
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
                try:
                    res[int(dh / 2):-int(dh / 2) - dh % 2, int(dw / 2):-int(dw / 2) - dw % 2] = original_area
                except ValueError:
                    return None, im

                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

                return res, im

        return None, im

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

    def __init__(self, grid: np.ndarray, svmModel: DSVM = None):
        self.digits = Digits(grid, model=svmModel)
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


def liveSolving():
    model_svm = DSVM()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.get()
        if not ret:
            break
        frame: np.ndarray = cv2.flip(frame, 1)
        sud = Sudoku(frame, svmModel=model_svm)
        to_show = cv2.warpPerspective(sud.sol_grid, sud.digits.cells.prep.original_M,
                                      (sud.digits.cells.prep.image.shape[1],
                                       sud.digits.cells.prep.image.shape[0]),
                                      dst=sud.digits.cells.prep.image, borderMode=cv2.BORDER_TRANSPARENT,
                                      flags=cv2.WARP_INVERSE_MAP)
        cv2.imshow('solved', to_show)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()
