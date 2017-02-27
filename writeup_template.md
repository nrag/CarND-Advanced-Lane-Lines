##Advanced Lane Finding Project
---

**The goals / steps of this project are the following:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistorted_test1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/gradient_test1.jpg "Binary Example"
[image4]: ./examples/birdseye_straight_lines1.jpg "Warp Example"
[image5]: ./examples/lane_test1.jpg "Fit Visual"
[image6]: ./examples/lane_test1.jpg "Output"
[video1]: ./project_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

--
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You're reading it! To look at the E2E implementation, open the IPython file `./scripts/Pipeline.ipynb`.

###Camera Calibration
####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in `./scripts/calibrate.py` (also look at `./scripts/Pipeline.ipynb`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Lane detection Pipeline (single images)

####Distortion correction
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####Color and Gradient thresholding
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `./scripts/gradient.py`):

*Absolute horizontal Sobel operator on the grayscale image
*Absolute vertical Sobel operator on the grayscale image
*Sobel operator in both horizontal and vertical directions and calculate its magnitude
*Sobel operator to calculate the direction of the gradient
*Convert the image from RGB space to HLS space, and threshold the S channel
*Combine the above binary images to create the final binary image

Here's an example of my output for this step.  (note: this is not actually from one of the test images)
![alt text][image3]

####Perspective transform

The code for my perspective transform (see `./scripts/perspective.py`) includes a function called `warp()`, which appears in lines 20 through 23 in the file `./scripts/perspective.py`. The `Perspective` class hardcodes the source and destination points and builds a transformation matrix based on these points. The `warp()` function takes as inputs an image (`img`), I chose the hardcode the source and destination points in the following manner:

```
        src = np.float32(
                  [[595, 450],
                   [685, 450],
                   [218, 720],
                  [1085, 720]])
                  
        dst = np.float32(
                   [[300, 0],
                    [980, 0],
                    [300, 720],
                   [980, 720]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 300, 0        | 
| 685, 450      | 980, 720      |
| 218, 720      | 300, 720      |
| 1085, 720     | 980, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####Lane identification and polynomial fit
To identify the lanes I did the following:
* I first took a histogram along all the columns in the lower half of the image like this
* With this histogram I added up the pixel values along each column in the image to determine the x-position of the lane in the base of the image.
* I  use that as a starting point a sliding window search, placed around the line centers, to find and follow the lines up to the top of the frame.
* If a lane has already been detected in the previous frame, I search in a margin around the previous line position

The code can be found in functions `detect()` `adjust()` in the file `./scripts/findlanes.py`
The output of the lane identification looks like this:

![alt text][image5]

####Radius of curvature
I calculated the radius of curvature for each line according to formulas presented here, using the polynomial fit for the left and right lane lines identified in the previous steps. I also converted the distance units from pixels to meters, assuming 30 meters per 720 pixels in the vertical direction, and 3.7 meters per 700 pixels in the horizontal direction.

I did this in function `curvature()` in my code in `./scripts/laneline.py`

####Lane Area
Using the outputs from the previous steps, I annotated the image as follows:

* I created a blank image, and draw our polyfit lines
* I filled the area between the lines
* Unwarped the perspective transform
* Overlaid the above annotation on the original image
* Added text to the original image to display lane curvature and vehicle offset

I implemented this step in lines # through # in my code in `laneline.py` in the function `highlight_lane()` and `annotate_image()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [video1](./project_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I implemented a basic version of the advanced lane finding algorithm. There are cases where the implementation is not doing a great job:
* In the video we can see that on oncrete roads, the lane detection does poorly
* In the video sometimes other vehicles in front of the car confuse the implementation into misrepresenting the lanes
* the challenge video which includes cracks are misidentified as lane lines 
