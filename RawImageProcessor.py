import os.path
import numpy as np
import cv2
import pickle
import glob
import matplotlib.pyplot as plt

CAMERA_CAL_PICTURE_PATH = 'Term1/CarND-Advanced-Lane-Lines/camera_cal/'
CAMERA_CAL_PICTURE_NAMES = 'calibration*.jpg'
CAMERA_CAL_PICKLE_PATH = 'Term1/CarND-Advanced-Lane-Lines/'
CAMERA_CAL_PICKLE_NAME = 'CarND-Advanced-Lane-Lines.pkl'

# chessboard grid size
GRID_SIZE = (9, 6)

class RawImageProcessor:

    def __init__(self):
        self.__calibrated = False
        self.__mtx = None
        self.__dist = None


    def __calibrateCamera(self, images, grid_size = GRID_SIZE):
        objpoints = []
        imgpoints = []
        image_size = (0,0)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

        for filename in images:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            image_size = gray.shape

            ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
            else:
                print("Unable to find required number of corners in " + filename)

        ret, self.__mtx , self.__dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        self.__calibrated = True


    def getCameraCalibration(self):
        if not os.path.isfile(CAMERA_CAL_PICKLE_PATH + CAMERA_CAL_PICKLE_NAME):
            print("No saved calibration. Calibrating now...")

            path = CAMERA_CAL_PICTURE_PATH + CAMERA_CAL_PICTURE_NAMES
            images = glob.glob(path)   
            self.__calibrateCamera(images, GRID_SIZE)
            print("Calibration complete")

            pickle.dump({'mtx': self.__mtx , 'dist': self.__dist}, open(CAMERA_CAL_PICKLE_PATH + CAMERA_CAL_PICKLE_NAME, 'wb'))
        else:        
            with open(CAMERA_CAL_PICKLE_PATH + CAMERA_CAL_PICKLE_NAME, mode='rb') as f:
                calibration = pickle.load(f)
                self.__mtx = calibration['mtx']
                self.__dist = calibration['dist']
                self.__calibrated = True
       
    # Corrects image distortion
    def undistort(self, image):
        if not self.__calibrated:
            self.getCameraCalibration()
        
        return cv2.undistort(image, self.__mtx, self.__dist, None, self.__mtx)


    # s_thresh=(170, 255), sx_thresh=(20, 100)
    def binary_pipeline(self, img, s_thresh=(250, 255), sx_thresh=(50, 100), p=99.9):

        # Convert to HLS color space and separate the s channel
        hls = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2HLS), (5, 5), 0)
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]
        
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Extract pixels that may be covered by shadow
        highlights = cv2.inRange(img[:, :, 0], int(np.percentile(img[:, :, 0], p) - 30), 255)

        # Extract yellow
        yellow = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), (21, 49, 148), (40, 255, 255))

        combined_binary = np.zeros_like(sxbinary)
        # combined_binary[(s_binary == 1) | (sxbinary == 1) | (yellow == 255) | (highlights == 255)] = 1
        combined_binary[(sxbinary == 1) | (yellow == 255) | (highlights == 255)] = 1

        # Ignore anything outside the region where lanes are expected to be (i.e. a trapezoidal shape)
        return self.__region_of_interest(combined_binary)


    # Applies an image mask. Only keeps the region of the image defined by the polygon 
    # formed from `vertices`. The rest of the image is set to black
    def __region_of_interest(self, image):
        MASK_X_PAD = 100
        MASK_Y_PAD = 95
        imshape = image.shape
        vertices = np.array([[(0, imshape[0]),                                            # bottom left
                            (imshape[1] / 2 - MASK_X_PAD, imshape[0] / 2 + MASK_Y_PAD),   # top left
                            (imshape[1] / 2 + MASK_X_PAD, imshape[0] / 2 + MASK_Y_PAD),   # top right
                            (imshape[1], imshape[0])]],                                   # bottom right
                            dtype=np.int32)

        mask = np.zeros_like(image)

        # Fill pixels inside the polygon defined by "vertices" with the fill color (i.e. white)
        cv2.fillPoly(mask, vertices, 255)

        # Return the image only where mask pixels are nonzero
        return cv2.bitwise_and(image, mask)
        


    