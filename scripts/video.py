import numpy as np
import cv2

from calibrate import CalibrateCamera
from gradient import ColorAndGradient
from perspective import Perspective
from findlanes import LaneDetection
from laneline import LaneLine
from moviepy.editor import VideoFileClip

class VideoLaneDetection:
    def __init__(self, calibration_path, video_file, output_file):
        self.calibrate = CalibrateCamera(calibration_path)
        self.gradient = ColorAndGradient(channel = 's', grad_channel='gray')
        self.perspective = Perspective()
        self.lane_detection = LaneDetection()

        self.video_file = video_file
        self.output_file = output_file

        self.window_size = 5  # how many frames for line smoothing
        self.previous_lanes = []
        self.detected = False  # did the fast line fit detect the lines?

    def add_lane(self, lane):
        """
        Gets most recent line fit coefficients and updates internal smoothed coefficients
        fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
        """
        # Coefficient queue full?
        full = len(self.previous_lanes) >= self.window_size
        self.previous_lanes.append(lane)
        if full:
          _ = self.previous_lanes.pop(0)

    def get_averaged_lane(self):
        left_A, right_A = [], []
        left_B, right_B = [], []
        left_C, right_C = [], []

        for lane in self.previous_lanes:
            left_A.append(lane.left_fit[0])
            left_B.append(lane.left_fit[1])
            left_C.append(lane.left_fit[2])
            right_A.append(lane.right_fit[0])
            right_B.append(lane.right_fit[1])
            right_C.append(lane.right_fit[2])

        avg_left_fit = [np.mean(left_A), np.mean(left_B), np.mean(left_C)]
        avg_right_fit = [np.mean(right_A), np.mean(right_B), np.mean(right_C)]

        avg_lane = LaneLine()
        avg_lane.left_fit = avg_left_fit
        avg_lane.right_fit = avg_right_fit
        avg_lane.left_lane_inds = self.previous_lanes[-1].left_lane_inds
        avg_lane.right_lane_inds = self.previous_lanes[-1].right_lane_inds
        avg_lane.nonzerox = self.previous_lanes[-1].nonzerox
        avg_lane.nonzeroy = self.previous_lanes[-1].nonzeroy
        avg_lane.image_shape = self.previous_lanes[-1].image_shape

        return avg_lane
      
    def detect_lane(self, img):
        # Undistort, threshold, perspective transform
        undistorted = self.calibrate.undistort(img)
        color_binary = self.gradient.filterAndThreshold(undistorted)
        binary_warped = self.perspective.warp(color_binary)

        # Perform polynomial fit
        if not self.detected:
            lane = self.lane_detection.detect(binary_warped)
            self.add_lane(lane)
            self.detected = True  # slow line fit always detects the line
        else:  # implies detected == True
            previous_lane = self.get_averaged_lane()
            lane = self.lane_detection.adjust(binary_warped, previous_lane)
            if lane is not None:
                self.add_lane(lane)
            else:
                self.detected = False

        # Perform final visualization on top of original undistorted image
        avg_lane = self.get_averaged_lane()
        highlighted = avg_lane.highlight_lane(undistorted, self.perspective)
        annotated = avg_lane.annotate_image(highlighted)

        return annotated


    def annotate_video(self):
        """ Given input_file video, save annotated video to output_file """
        video = VideoFileClip(self.video_file)
        annotated_video = video.fl_image(self.detect_lane)
        annotated_video.write_videofile(self.output_file, audio=False)
