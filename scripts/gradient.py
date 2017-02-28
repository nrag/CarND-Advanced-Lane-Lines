import numpy as np
import cv2

class ColorAndGradient():
    def __init__(self, channel = 's', grad_channel='gray', channel_threshold = (100,255), grad_threshold=(50,255)):
        self.channel = channel
        self.grad = grad_channel
        self.ksize = 3
        self.mag_thresh = (50,255)
        self.dir_thresh = (0.7, 1.3)
        self.channel_threshold = channel_threshold
        self.grad_threshold = grad_threshold

    def getChannel(self, img, c):
        if c == 'h':
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
            return hsv[:, :, 0]

        if c == 's':
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
            return hsv[:, :, 1]

        if c == 'v':
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
            return hsv[:, :, 2]

        if c == 'l':
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
            return hsv[:, :, 1]

        if c == 'gray':
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel = 3, thresh=(0,255)):
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            
        # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    def mag_sobel_thresh(self, gray, sobel_kernel=3, thresh=(0, 255)):
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            
        # 3) Calculate the magnitude 
        abs_sobel = np.sqrt(sobelx**2 + sobely**2)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi/2)):
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
            
        # 3) Calculate the magnitude 
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        grad_direction = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(grad_direction)
        binary_output[(grad_direction >= thresh[0]) & (grad_direction <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    def combined_threshold(self, image):
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=self.ksize, thresh=self.grad_threshold)
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=self.ksize, thresh=self.grad_threshold)
        mag_binary = self.mag_sobel_thresh(image, sobel_kernel=self.ksize, thresh=self.mag_thresh)
        dir_binary = self.dir_threshold(image, sobel_kernel=15, thresh=self.dir_thresh)
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

    def filterAndThreshold(self, img):
        # Convert to HSV color space and separate the V channel
        img_channel = self.getChannel(img, self.channel) 
        grad_channel = self.getChannel(img, self.grad)

        # Sobel 
        grad_binary = self.combined_threshold(grad_channel)
           
        # Threshold color channel
        img_binary = np.zeros_like(img_channel)
        img_binary[(img_channel >= self.channel_threshold[0]) & (img_channel <= self.channel_threshold[1])] = 1

        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        color_binary = np.zeros_like(grad_binary)
        color_binary[(img_binary == 1) | (grad_binary == 1)] = 1
        return color_binary
