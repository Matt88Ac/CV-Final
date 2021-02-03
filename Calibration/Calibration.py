import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class ChessCalibrator:

    def __init__(self, plot_process=False, x=11, y=12):
        images = [f'Calibration/calib_example/Image{i}.tif' for i in range(1, 26)]

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.000001)

        self.objp = np.zeros((x * y, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.

        for i, fname in enumerate(images):
            img = cv2.imread(fname)
            if len(img.shape) != 2:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            print(f"Processing {i+1}/26 image...")
            ret, corners = cv2.findChessboardCorners(gray, (x, y), None)
            if ret:
                self.objpoints.append(self.objp)

                corners2 = cv2.cornerSubPix(gray, corners, (x, y), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                if plot_process:
                    if len(img.shape) != 2:
                        img = cv2.drawChessboardCorners(img, (x, y), corners2, ret)
                    else:
                        img = cv2.drawChessboardCorners(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (x, y), corners2, ret)
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.show()

        self.imgpoints = np.array(self.imgpoints)
        self.objpoints = np.array(self.objpoints)
        self.ret, self.mtx, self.dist_coeff, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints,
                                                                                          self.imgpoints,
                                                                                          gray.shape[::-1], None, None)
        self.tvecs = np.array(self.tvecs)
        self.rvecs = np.array(self.rvecs)

        # self.RT = np.zeros((len(self.objpoints), 3, 4))
        # self.RT[:, :, :3] = self.rvecs.copy()
        # self.RT[:, :, 3] = self.tvecs.reshape((len(self.objpoints), 3))

        self.shape = gray.shape

    def __plotError(self):
        h, w = self.shape
        fig = plt.figure()
        a = np.array([[0, 0, 0]])
        b = np.array([[w, 0, 0]])
        c = np.array([[w, h, 0]])
        d = np.array([[0, h, 0]])
        ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')

        S = np.zeros((4, 4))
        np.fill_diagonal(S, [w, h, 1, 1])

        colors = ['black', 'green', 'navy', 'orange']

        xx = np.zeros((1, 3))
        yy = xx.copy()
        zz = xx.copy()

        for i in range(len(self)):
            R = np.eye(4)
            T = R.copy()
            R[:3, :3] = Rotation.from_rotvec(self.rvecs[i, :, 0]).as_matrix()
            T[:3, 3] = self.tvecs[i, :, 0]
            Xc = T * R * S
            print(Xc)
            print(self.objpoints[i].shape)
            return
            # ax.plot_wireframe(xx, yy, zz, color=colors[i])
            # plt.show()

            # x_c = np.dot(self.rvecs[i], letter) + self.tvecs[i]
            # xx[j] = np.squeeze(x_c[:, 0])
            # yy[j] = np.squeeze(x_c[:, 1])
            # zz[j] = np.squeeze(x_c[:, 2])

    def undisort(self, image: np.ndarray) -> np.ndarray:
        print('Calibrating image...')
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist_coeff, (w, h), 0, (w, h))
        dst = cv2.undistort(image, self.mtx, self.dist_coeff, None, newcameramtx)

        x, y, w, h = roi
        print('Done Calibrating!')

        return dst[y:h + y + 1, x:w + x + 1]

    def __len__(self):
        return self.objpoints.shape[0]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.undisort(image)


# C = ChessCalibrator(x=13, y=12, plot_process=True)
