import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from ML_Models import DigitsSVM as DSVM
from SudokuSolver import Sudoku
import warnings
from Calibration.Calibration import ChessCalibrator

warnings.filterwarnings("ignore")


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
    calibrator = ChessCalibrator()

    def findAllAreas(feed: np.ndarray):
        feed = calibrator.undisort(image=feed)

        def PreProcess():
            imag = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)

            imag = cv2.GaussianBlur(imag, (5, 5), 0)
            # imag = cv2.fastNlMeansDenoising(imag, None)
            imag = cv2.adaptiveThreshold(imag, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            imag = cv2.bitwise_not(imag, imag)

            kernel = np.ones((2, 2), np.uint8)

            imag = cv2.dilate(imag, kernel, iterations=1)

            return imag

        gray = PreProcess()
        contours, h = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        x, y, w, h = cv2.boundingRect(contours[0])

        cv2.rectangle(feed, (x, y), (x + w, y + h), (35, 135, 200), thickness=3)

        return feed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_frame = findAllAreas(frame.copy())
        success = False
        try:
            sud = Sudoku(frame, svmModel=model_svm)
            new_frame = cv2.warpPerspective(sud.sol_grid, sud.digits.cells.prep.original_M,
                                            (sud.digits.cells.prep.image.shape[1],
                                             sud.digits.cells.prep.image.shape[0]),
                                            dst=sud.digits.cells.prep.image, borderMode=cv2.BORDER_TRANSPARENT,
                                            flags=cv2.WARP_INVERSE_MAP)
            success = True
        except ValueError:
            pass

        except TypeError:
            pass
        if not success:
            cv2.imshow('Looking for Sudoku grid...', new_frame)
            key = cv2.waitKey(1)

            if key == 27:  # exit on ESC
                break

        else:
            cv2.imshow('Found the solution! press esc to continue', new_frame)
            key = cv2.waitKey(0)

            if key == 27:  # exit on ESC
                cv2.destroyAllWindows()
                continue

    cap.release()
    cv2.destroyAllWindows()
