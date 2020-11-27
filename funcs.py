import numpy as np
import cv2


def imagePreProcessing(image: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.fastNlMeansDenoising(img, None)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    img = cv2.bitwise_not(img, img)

    kernel = np.ones((2, 2), np.uint8)

    img = cv2.dilate(img, kernel, iterations=1)

    return img


def runSVM(train, labels):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(train.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.int32))
    return svm


def findArea(image: np.ndarray):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestContours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    epsilon = 0.015 * cv2.arcLength(largestContours[0], True)
    approx = cv2.approxPolyDP(largestContours[0], epsilon, True)

    return approx


def four_point_transform(image: np.ndarray, pts: np.ndarray):
    image = image.copy()

    def order_points(points: np.ndarray):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect1 = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect1[0] = points[np.argmin(s)]
        rect1[2] = points[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(points, axis=1)
        rect1[1] = points[np.argmin(diff)]
        rect1[3] = points[np.argmax(diff)]

        # return the ordered coordinates
        return rect1

    pts = pts.reshape(4, 2)
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
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
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), borderMode=cv2.BORDER_TRANSPARENT)

    # return the warped image
    return warped, M


def imageToSudokuCells(image: np.ndarray) -> np.ndarray:
    cells = [np.array_split(row, 9, axis=1) for row in np.array_split(image, 9)]
    cells = np.asarray(cells).reshape(81, 1)

    return cells


def extract_digit(image, kernel_size: tuple = (5, 5)):
    original = image.copy()
    image = image.copy()

    o_h, o_w = image.shape

    # shape of dataset item
    r_w = 50
    r_h = 50

    if o_h > r_h or o_w > r_w:
        image = cv2.resize(image, dsize=(r_h - 2, r_w - 2))
        o_h, o_w = image.shape

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    for row in range(o_h):
        if image[row, 0] == 255:
            cv2.floodFill(image, None, (0, row), 0)
        if image[row, o_w - 1] == 255:
            cv2.floodFill(image, None, (o_w - 1, row), 0)

    for col in range(o_w):
        if image[0, col] == 255:
            cv2.floodFill(image, None, (col, 0), 0)
        if image[o_h - 1, col] == 255:
            cv2.floodFill(image, None, (col, o_h - 1), 0)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        original_area = original[y:y + h, x:x + w]
        n_h, n_w = original_area.shape

        # validate contours is number and not noise
        if n_h > (o_h / 5) and n_w > (o_w / 5):
            res = np.zeros((r_h, r_h))
            res[int((r_h - h) / 2):-int((r_h - h) / 2) - (r_h - h) % 2,
            int((r_w - w) / 2):-int((r_w - w) / 2) - (r_w - w) % 2] = original_area

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

            return res, image

    return None, image


def drawOnSodukoCells(cells, digits, original_digits, color=(0, 255, 0)):
    res = []

    for i in range(len(cells)):
        cell = cells[i][0]
        digit = digits[i]
        is_original = original_digits[i] != 0

        # draw only solution digits - not originals
        if not is_original:
            res.append(drawDigit(cell, digit, color))
        else:
            res.append(cell)

    return np.asarray(res)


def drawDigit(image, digit, color=(0, 255, 0)):
    h, w, _ = image.shape

    size = w / h
    offsetTop = int(h * 0.75 * size)
    offsetLeft = int(w * 0.25 * size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(image.copy(), str(int(digit)), (offsetLeft, offsetTop), font, size, color, 2, cv2.LINE_AA)
    return img


def is_safe(array, r, c, num):
    def already_used(n, a):
        return n in a

    box_r = r - r % 3
    box_c = c - c % 3
    box = already_used(num, array[box_r:box_r + 3, box_c:box_c + 3])

    return not already_used(num, array[r]) and not already_used(num, array[:9, c:c + 1]) and not box


def solve(arr: np.ndarray):
    def next_cell(a: np.ndarray) -> tuple:
        if arr[0][0] == 0:
            return 0, 0
        (r, c) = np.unravel_index((a == 0).argmax(), a.shape)
        if r != 0 or c != 0:
            return r, c

        return -1, -1

    row, col = next_cell(arr)

    # solved
    if row == col == -1:
        return True

    for num in range(1, 10):
        if is_safe(arr, row, col, num):
            arr[row][col] = num

            # solved
            if solve(arr):
                return True

            # current num is not true
            # try again
            arr[row][col] = 0

    # no solution
    return False


def predict(image, hog, svm):
    image = cv2.resize(image, dsize=(50, 50))
    im = image.reshape(50, 50).astype(np.uint8)
    im_h = hog.compute(im)
    svm_res = svm.predict(np.array([im_h]))

    return svm_res


def should_try(sudoku: np.ndarray):
    # if has at least 2 digit in each sudoku box (3x3)
    for i in range(3):
        for j in range(3):
            box = sudoku[i * 3:i * 3 + 3, j * 3:j * 3 + 3]
            if len(box[box != 0]) < 2:
                return False

    # if current sudoku is valid
    # check if each cell value (from SVM) is valid (no duplicate, etc..)
    for i in range(9):
        for j in range(9):
            arr = sudoku.copy()
            num = arr[i, j]
            if num != 0:
                arr[i, j] = 0
                if not is_safe(arr, i, j, num):
                    return False

    return True


def find_sudoku_solution(sudoku: np.ndarray):
    solved = sudoku.copy()

    if should_try(solved):
        if solve(solved):
            return solved

    return None


def sudokuCellsToImage(cells, like_image):
    img = np.zeros_like(like_image)

    h = 0
    w = 0
    cell = cells[0]

    for i in range(9):
        for j in range(9):
            index = i * 9 + j
            cell = cells[index]

            img[h: h + cell.shape[0], w: w + cell.shape[1]] = cell
            w += cell.shape[1]

        w = 0
        h += cell.shape[0]

    return img
