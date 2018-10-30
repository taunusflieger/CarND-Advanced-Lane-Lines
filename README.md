## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Distorted"
[image1]: ./calibration1_output.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Distorted"
[image3]: ./output_images/test1_undist_output.jpg  "Undistorted"
[image4]: ./output_images/test1_output.jpg "Binary Example"
[image5]: ./output_images/test2_output.jpg "Binary Example"
[image6]: ./output_images/test5_output.jpg "Binary Example"
[image7]: ./output_images/video_frame_output.jpg "Video Frame"

[video1]: ./project_video_output.mp4 "Video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is in the file called `RawImageProcessor.py`. The class `RawImageProcessor`contains the code to calculate or load the camera calibration parameter and provide the functions to undistore an image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


Distorted image:
![alt text][image0]
Undistorted image:
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

This step is part of the `LineFinder`class in the file `LineFinder.py`. Within the method used to process a single image (`process_image`) or to process a video frame (`process_video_frame`), the first step within the image processing pipeline is the removal of image distortion using the method `rawImageProcessor.undistort(orig_image)`to remove image distortion. This method is using the calibration data previously generated using the chessboard images. The following images demonstrating the result of this process:

Distorted image:
![alt text][image2]
Undistorted image:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of sobel and gradient combined with recovery of shadow elements and detection of the yellow line parts to generate the binary image result. This is done within the method `RawImageProcessor.binary_pipeline()`located in the file `RawImageProcessor.py`. At the end of the process a clipping mask will be used to extract the area of interest of the image in which the lane lines are located.

Here's an example of my output for this step combined with the final processed image visualizing the detected lanes


Processed binary image (middle image in the lower line)
![alt text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
The perspective transformation is implemented in the `LaneFinder` class in the file `LaneFinder.py` After generating the binary image in either `process_video_frame()`or `process_image()` the transformation is executed. It is implemented within the method `__warp()` using `cv2.warpPerspective`. The transformation coefficients are calculated during the initialization of the class `LaneFinder` using hardcoded points:  

```python
# Define the source points
self.src = np.float32([[ 300, Y_BOTTOM],    # bottom left
                [ 580, Y_HORIZON],          # top left
                [ 730, Y_HORIZON],          # top right
                [1100, Y_BOTTOM]])          # bottom right

# Define the destination points
self.dst = np.float32([
    (self.src[0][0]  + OFFSET, Y_BOTTOM),
    (self.src[0][0]  + OFFSET, 0),
    (self.src[-1][0] - OFFSET, 0),
    (self.src[-1][0] - OFFSET, Y_BOTTOM)])
```
In the lower line the right image shows the perspective transformation result (together with the visualization of the lane detection)
![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane detection utilizes two approaches
1. Sliding Window Search
2. Search from Prior

For single images only approach 1) is used. For processing the video stream a combination of both approaches is used. First sliding window search is used to calculate the first two polynoms. These polynomes (one for the right line and one for the left line) are in the following frame to search only in an area close to where the first lines where found. This approach is significantly faster than processing each frame with the sliding window approach. In case that the `Search from Prior approach` does not generate a suitable result (no lines found, lines are not parallel or distance between lines is out of a defined boundary), the same frame is processed again with the `Sliding Window Search``

The Sliding Window Search is implemented in the `LaneFinder.__detect_lanes()` and `LaneFinder.__sliding_window()`method. The core is in the later method. The general approach is to use a histogram to identify the starting area for the sliding window search of the left and right line. Within the defined windows non-zero pixes are identified which will than used as an input to define a polynom of 3rd order which represents the line. The coefficents of the polynom are stored and used to draw the polyshape which visualizes the boundaries of the detected lane.

Example of the `Sliding Window Search`in the lower right image
![alt text][image6]

Example of the `Search from Prior` in the lower right image
![alt text][image7]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calculated in the method `Line.calc_curvature_radius()`in file `Line.py` and is using the approach which has been outlined in the training session.





#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `LaneFinder.__plot_lane` in file `LaneFinder.py`. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The quality of the thresholded binary image is key for successfully identifying lane lines. A lot of optimization can be done here, but there is no one-fits-it-all solution. Obviously road conditions and light are influencing the result. Parameters which optimize a certain condition very well could work rather bad in other condition. Having a more adaptive filter solution will increase probability to detect the lane lines.

I've not spend time to further optimize for more challenging scenarios - narrow curves, different light conditions. 
To get better results for the `challenge_video.mpg`, the thresholded binary image pipeline needs to be further optimized and maybe more adaptive.
