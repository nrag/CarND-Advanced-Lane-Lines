import cv2
import numpy as np

class LaneLine():
    def __init__(self):
        self.left_fit = None
        self.left_lane_inds = None
        self.right_fit = None
        self.right_lane_inds = None
        self.nonzerox = None
        self.nonzeroy = None
        self.image_shape = None

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension

    def curvature(self):
        y_eval = 719  # 720p video/image, so last (lowest on screen) y index is 719

        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]
        ploty = np.linspace(0, self.image_shape[0]-1, self.image_shape[0])

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        return left_curverad, right_curverad

    def visualize(self, binary_warped):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + right_fit[2]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], nonzerox[self.right_lane_inds]] = [0, 0, 255]
        return out_img, left_fitx, right_fitx

    def visualize_band(self, binary_warped):
        """
        Visualize the predicted lane lines with margin, on binary warped image
        save_file is a string representing where to save the image (if None, then just display)
        """

        # Create an image to draw on and an image to show the selection window
        out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        margin = 100  # NOTE: Keep this in sync with *_fit()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        return result

    def vehicle_offset(self, image):
        # Calculate vehicle center offset in pixels
        bottom_y = image.shape[0] - 1
        bottom_x_left = self.left_fit[0]*(bottom_y**2) + self.left_fit[1]*bottom_y + self.left_fit[2]
        bottom_x_right = self.right_fit[0]*(bottom_y**2) + self.right_fit[1]*bottom_y + self.right_fit[2]
        vehicle_offset = image.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

        # Convert pixel offset to meters
        vehicle_offset *= self.xm_per_pix

        return vehicle_offset

    def highlight_lane(self, undistorted, perspective):
        """
        Final lane line prediction visualized and overlayed on top of original image
        """
        # Generate x and y values for plotting
        ploty = np.linspace(0, undistorted.shape[0]-1, undistorted.shape[0])
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

        #warp_zero = np.zeros_like(warped).astype(np.uint8)
        #color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = perspective.unwarp(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        return result

    def annotate_image(self, image):
        # Annotate lane curvature values and vehicle offset from center
        left_curvead, right_curvead = self.curvature()
        avg_curve = (left_curvead + right_curvead)/2
        label_str = 'Radius of curvature: %.1f m' % avg_curve
        result = cv2.putText(image, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

        vehicle_offset = self.vehicle_offset(image)
        label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
        result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

        return result
