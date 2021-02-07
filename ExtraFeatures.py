import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from ML_Models import DigitsSVM as DSVM


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