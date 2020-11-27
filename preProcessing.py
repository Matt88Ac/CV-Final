import numpy as np
import cv2


class preProcessing:

    def __init__(self, image: np.ndarray):
        self.image: np.ndarray = image.copy()

        def runPP(matrix: np.ndarray) -> np.ndarray:
            img = cv2.cvtColor(matrix.copy(), cv2.COLOR_BGR2GRAY)

            img = cv2.GaussianBlur(img, (9, 9), 0)
            img = cv2.fastNlMeansDenoising(img, None)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            img = cv2.bitwise_not(img, img)

            kernel = np.ones((2, 2), np.uint8)

            img = cv2.dilate(img, kernel, iterations=1)

            return img

        self.grayIm = runPP(image)
