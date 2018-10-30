import numpy as np
import cv2
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Line import Line
from RawImageProcessor import RawImageProcessor

XM_PER_PIXEL = 3.7 / 700            # meters per pixel in x dimension
YM_PER_PIXEL = 30.0 / 720           # meters per pixel in y dimension
IMAGE_MAX_Y = 719                   # bottom of the image (image/frame resolution is fixed at 1280/720)

# HYPERPARAMETERS
# Choose the number of sliding windows
NWINDOWS = 9
PEAK_MIN_WIDTH = 70                 # used by find_peaks_cwt to detect peaks
PEAK_MAX_WIDTH = 100                # as above
PEAK_SLIDING_WINDOW_NUM = 9         # number of sliding windows to use
PEAK_SLIDING_WINDOW_WIDTH = 50      # number of pixels a sliding window extends to the left and right of its centre

MIN_POINTS_REQUIRED = 3
OFFSET = 260
Y_HORIZON = 470
Y_BOTTOM = 720

class LaneFinder:
    def __init__(self, parallel_thresh=(0.0003, 0.55), lane_dist_thresh=(300, 460), n_frames=1):
        self.n_frames = n_frames
        self.left_line = None
        self.right_line = None   
        self.curvature_radius_left = 0
        self.curvature_radius_right = 0
        self.center_poly = None
        self.offset = 0   
        self.diag_frame_count = 0

        self.parallel_thresh = parallel_thresh
        self.lane_dist_thresh = lane_dist_thresh

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

        # Compute the perspective transform
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)


   # Detect lane lines using the sliding window approach for the first frame
    def process_image(self, orig_image, diag=False):
        rawImageProcessor = RawImageProcessor()

        # 1. Correct image distortion
        image_undist = rawImageProcessor.undistort(orig_image)

        # 2. Create thresholded binary image
        image_thres = rawImageProcessor.binary_pipeline(image_undist)
        
        # 3. Apply perspective transform to create a bird's-eye view 
        image_warp = self.__warp(image_thres)

        # 4. Detect and plot the lane
        image_final, out_img = self.__detect_lanes(image_warp, image_undist)

        if (diag==False):
            return image_final

        # 3. Optional: return the diagnostic image instead
        diag_image = self.__create_diag_output(image_undist, image_thres, out_img, image_final)

        return diag_image


    # Detect lane lines using the sliding window approach for the first frame
    # After the first frame the algorith will search in a margin around the previous line position.
    # This appraoch speeds up video processing
    def process_video_frame(self, frame):
        rawImageProcessor = RawImageProcessor()

        # 1. Correct image distortion
        image_undist = rawImageProcessor.undistort(frame)

        # 2. Create thresholded binary image
        image_thres = rawImageProcessor.binary_pipeline(image_undist)
        
        # 3. Apply perspective transform to create a bird's-eye view 
        image_warp = self.__warp(image_thres)

        # 4. Detect and plot the lane
        if (self.left_line is not None and self.right_line is not None and self.left_line.best_fit_exits() 
            and self.right_line.best_fit_exits()):
            # Use existing polynomes as starting point
            lanes_found, image_final, out_img = self.__search_around_poly(image_warp, image_undist)
            if not lanes_found:
                # Fallback, start from scratch at current frame
                image_final, out_img = self.__detect_lanes(image_warp, image_undist)     
        else:
            # Start from scratch
            image_final, out_img = self.__detect_lanes(image_warp, image_undist)

        out_img = self.__create_diag_output(image_undist, image_thres, out_img, image_final)
        if self.diag_frame_count == 12:
            mpimg.imsave('Term1/CarND-Advanced-Lane-Lines/output_images/video_frame_output.jpg', out_img)
        self.diag_frame_count += 1
        return out_img

  
    # Uses a sliding window algorithm to identify pixels within the image
    # which are part of a lane line
    def __sliding_window(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 90
        # Set minimum number of pixels found to recenter window
        minpix = 60

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img


    # Determines if detected lane lines are plausible lines based on curvature and distance
    def __check_lane_quality(self, left, right):
        if len(left[0]) < MIN_POINTS_REQUIRED or len(right[0]) < MIN_POINTS_REQUIRED:
            return False
        else:
            new_left = Line(detected_y=left[0], detected_x=left[1])
            new_right = Line(detected_y=right[0], detected_x=right[1])

            parallel_check = new_left.check_lines_parallel(new_right, threshold=self.parallel_thresh)
            dist = new_left.distance_between_lines(new_right)
            dist_check = self.lane_dist_thresh[0] < dist < self.lane_dist_thresh[1]

            return parallel_check & dist_check

    # Check detected lines against each other and against previous frames' lines to ensure they are valid lines
    def __validate_lines(self, left_points, right_points):
        left_detected = False
        right_detected = False

        if self.__check_lane_quality(left_points, right_points):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.__check_lane_quality(left_points, (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.__check_lane_quality(right_points, (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected


    # Calculate the line curvature (in meters)
    def __calc_curvature_radius(self, fit_cr):
        y = np.array(np.linspace(0, IMAGE_MAX_Y, num=10))
        x = np.array([fit_cr(x) for x in y])
        y_eval = np.max(y)

        fit_cr = np.polyfit(y * YM_PER_PIXEL, x * XM_PER_PIXEL, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval / 
                    2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curverad


    # Perspective transformation to bird's eye view
    def __warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    # Reverse perspective transformation
    def __warp_inv(self, image):
        return cv2.warpPerspective(image, self.Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


    # Detect lane lines using the sliding window approach 
    def __detect_lanes(self, image_warp, image_undist):
        left_detected = right_detected = False

        # Get lane pixels for left and right lane using sliding window approach
        # out_img contains visualization of the process
        leftx, lefty, rightx, righty, out_img = self.__sliding_window(image_warp)

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        if not left_detected or not right_detected:
            left_detected, right_detected = self.__validate_lines((leftx, lefty ),
                                                                  (rightx, righty ))

        # Update line information
        if left_detected:
            if self.left_line is not None:
                self.left_line.update(y = leftx, x = lefty)
            else:
                self.left_line = Line(self.n_frames,
                                      detected_y = leftx,
                                      detected_x = lefty)

        if right_detected:
            if self.right_line is not None:
                self.right_line.update(y = rightx, x = righty)
            else:
                self.right_line = Line(self.n_frames,
                                       detected_y = rightx,
                                       detected_x = righty)

        # Draw the lane and add additional information
        if self.left_line is not None and self.right_line is not None:
            self.curvature_radius_left = self.__calc_curvature_radius(self.left_line.best_fit_poly)
            self.curvature_radius_right = self.__calc_curvature_radius(self.right_line.best_fit_poly)
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.offset = (image_undist.shape[1] / 2 - self.center_poly(IMAGE_MAX_Y)) * XM_PER_PIXEL

            image_undist = self.__plot_lane(image_undist, image_warp)  
        return image_undist, out_img


    # Another approach to detec lane lines. This one uses the information
    # about line location from the previous video frame to narrow down
    # the area of intest.
    def __search_around_poly(self, image_warp, image_undist):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 70
        out_img = None

        # Grab activated pixels
        nonzero = image_warp.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = self.left_line.best_fit
        right_fit = self.right_line.best_fit

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
            
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_detected, right_detected =self.__validate_lines((leftx, lefty), (rightx, righty))
        if left_detected and right_detected:
            self.left_line.update(y = leftx, x = lefty)
            self.right_line.update(y = rightx, x = righty)

            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            ploty = np.linspace(0, image_warp.shape[0]-1, image_warp.shape[0])
            left_fitx = (self.left_line.current_fit[0]*ploty**2 + self.left_line.current_fit[1]*ploty + 
                         self.left_line.current_fit[2])
            right_fitx = (self.right_line.current_fit[0]*ploty**2 + self.right_line.current_fit[1]*ploty + 
                          self.right_line.current_fit[2])

            out_img = np.dstack((image_warp, image_warp, image_warp))*255
            window_img = np.zeros_like(out_img)

            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

            # Plot the polynomial lines onto the image
            out_img = cv2.addWeighted(out_img, 0.9, window_img, 0.1, 0)
            cv2.polylines(out_img, np.int_([np.array([np.transpose(np.vstack([right_fitx, ploty]))])]),  False,  (255, 240, 0),  thickness=3)
            cv2.polylines(out_img, np.int_([np.array([np.transpose(np.vstack([left_fitx, ploty]))])]),  False,  (255, 240, 0),  thickness=3)

            ## End visualization steps ##

        
            # Draw the lane and add additional information
            if self.left_line is not None and self.right_line is not None:
                self.curvature_radius_left = self.__calc_curvature_radius(self.left_line.best_fit_poly)
                self.curvature_radius_right = self.__calc_curvature_radius(self.right_line.best_fit_poly)
                self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
                self.offset = (image_undist.shape[1] / 2 - self.center_poly(IMAGE_MAX_Y)) * XM_PER_PIXEL

                image_undist = self.__plot_lane(image_undist, image_warp)
                lanes_found = True
        else:
            lanes_found = False

        return lanes_found, image_undist, out_img


    # Plot the detected lane and information about curvature radius and car off-center 
    def __plot_lane(self, imgage_org, imgage_warp):
        warp_zero = np.zeros_like(imgage_warp).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        yrange = np.linspace(0, 720)
        fitted_left_x = self.left_line.best_fit_poly(yrange)
        fitted_right_x = self.right_line.best_fit_poly(yrange)

        pts_left = np.array([np.transpose(np.vstack([fitted_left_x, yrange]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([fitted_right_x, yrange])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank image back to original image space using inverse perspective matrix (Minv)
        newwarp = self.__warp_inv(color_warp)

        # Combine the result with the original image
        result = cv2.addWeighted(imgage_org, 1, newwarp, 0.3, 0)
                                                                               
        # Add information overlay rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        result_overlay = result.copy()
        cv2.rectangle(result_overlay, (10, 10), (410, 150), (255, 255, 255), -1)
        cv2.addWeighted(result_overlay, 0.7, result, 0.3, 0, result)

        lane_center_x = int(self.center_poly(IMAGE_MAX_Y))
        image_center_x = int(result.shape[1] / 2)
        offset_from_centre = (image_center_x - lane_center_x) * XM_PER_PIXEL               # in meters
        
        # Add curvature and offset information
        font_color = (60, 60, 60) # dark grey
        left_right = 'left' if offset_from_centre < 0 else 'right'
        cv2.putText(result, "{:>14}: {:.2f}m ({})".format("off-center", abs(offset_from_centre), left_right),
                    (24, 50), font, 0.8, font_color, 2)

        text = "{:.1f}m".format(self.curvature_radius_left)
        cv2.putText(result, "{:>15}: {}".format("Left curvature", text), (19, 90), font, 0.8, font_color, 2)

        text = "{:.1f}m".format(self.curvature_radius_right)
        cv2.putText(result, "{:>15}: {}".format("Right curvature", text), (15, 130), font, 0.8, font_color, 2)

        return result


    def __create_diag_output(self, image, preprocessed_image, warp_image, image_final):
        # Create a combined image with final result and intermidiate processing steps
        diag_output = np.zeros((1080, 1280, 3), dtype=np.uint8)

        # Main screen
        diag_output[0:720, 0:1280] = image_final

        # Three screens along the bottom
        diag_output[720:1080, 0:426] = cv2.resize(image, (426,360), interpolation=cv2.INTER_AREA)            # original frame
        
        color_thresh = np.dstack((preprocessed_image, preprocessed_image, preprocessed_image)) * 255
        diag_output[720:1080, 426:852] = cv2.resize(color_thresh, (426,360), interpolation=cv2.INTER_AREA)    # undistorted binary filtered
       
        if len(warp_image.shape) == 2:
            color_warp = np.dstack((warp_image, warp_image, warp_image)) * 255
        else:
            color_warp = warp_image
        diag_output[720:1080, 852:1278] = cv2.resize(color_warp, (426,360), interpolation=cv2.INTER_AREA)   # warped image

        return diag_output
