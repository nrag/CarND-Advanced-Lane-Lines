import cv2
import glob
import matplotlib.image as mpimg
import numpy as np


class CalibrateCamera():
    # TODO: Write a function that takes an image, object points, and image points
    # performs the camera calibration, image distortion correction and 
    # returns the undistorted image
    def __init__(self, path):
        images = glob.glob(path)
        
        objpoints = [] #3D points in real world space
        imgpoints = [] #2D points in image plane

        objp = np.zeros((6*8, 3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        for fname in images:
            img = mpimg.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret,corners = cv2.findChessboardCorners(gray, (8,6), None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

            img_size = (img.shape[1], img.shape[0])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.mtx = mtx
        self.dist = dist

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

