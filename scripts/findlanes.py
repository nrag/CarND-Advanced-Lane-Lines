import numpy as np
import cv2
import matplotlib.pyplot as plt

from laneline import LaneLine

class LaneDetection():
    def __init__(self):
        # Choose the number of sliding windows
        self.nwindows = 9 
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

    def detect(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[100:midpoint]) + 100
        rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/self.nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        lane = LaneLine() 
        lane.left_fit = left_fit
        lane.left_lane_inds = left_lane_inds
        lane.right_fit = right_fit
        lane.right_lane_inds = right_lane_inds
        lane.nonzerox = nonzerox
        lane.nonzeroy = nonzeroy
        lane.image_shape = binary_warped.shape

        return lane 

    def adjust(self, binary_warped, previous_lane):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (previous_lane.left_fit[0]*(nonzeroy**2) + 
                                       previous_lane.left_fit[1]*nonzeroy + 
                                       previous_lane.left_fit[2] - margin)) & 
                          (nonzerox < (previous_lane.left_fit[0]*(nonzeroy**2) + 
                                       previous_lane.left_fit[1]*nonzeroy + 
                                       previous_lane.left_fit[2] + self.margin))) 
        right_lane_inds = ((nonzerox > (previous_lane.right_fit[0]*(nonzeroy**2) + 
                                        previous_lane.right_fit[1]*nonzeroy + 
                                        previous_lane.right_fit[2] - margin)) & 
                           (nonzerox < (previous_lane.right_fit[0]*(nonzeroy**2) + 
                                        previous_lane.right_fit[1]*nonzeroy + 
                                        previous_lane.right_fit[2] + self.margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        min_inds = 10
        if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
            return None

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        lane = LaneLine() 
        lane.left_fit = left_fit
        lane.left_lane_inds = left_lane_inds
        lane.right_fit = right_fit
        lane.right_lane_inds = right_lane_inds
        lane.nonzerox = nonzerox
        lane.nonzeroy = nonzeroy
        lane.image_shape = binary_warped.shape

        self.previous_lane = lane
        self.previous_image = binary_warped 
        return lane
