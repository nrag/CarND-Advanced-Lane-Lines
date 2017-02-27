import cv2
import numpy as np

class Perspective:
    def __init__(self):
        self.dst = np.float32(
                [[300, 0], 
                 [980, 0],
                 [300, 720],
                 [980, 720]])
        self.src = np.float32(
                [[595, 450],
                 [685, 450],
                 [218, 720],
                 [1085, 720]])
        
        self.transform = cv2.getPerspectiveTransform(self.src, self.dst)
        self.inv_transform = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.transform, img_size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.inv_transform, img_size, flags=cv2.INTER_LINEAR)
