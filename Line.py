import numpy as np

XM_PER_PIXEL = 3.7 / 700                # meters per pixel in x dimension
YM_PER_PIXEL = 30.0 / 720               # meters per pixel in y dimension

IMAGE_MAX_Y = 719

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, n_frames=1, detected_x=None, detected_y=None):
        # was the line detected in the last iteration?
        self.detected = False  
        # Number of previous frames used to smooth the current frame
        self.n_frames = n_frames     
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        #self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  

        self.update(detected_x, detected_y)

    def best_fit_exits(self):
        return self.best_fit is not None 

    def update(self, x, y):
        self.allx = x
        self.ally = y

        # Fit a polynomial and smooth it using previous frames
        self.current_fit = np.polyfit(self.allx, self.ally, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)


    def check_lines_parallel(self, other_line, threshold=(0, 0)):
        diff_coeff_first = np.abs(self.current_fit[0] - other_line.current_fit[0])
        diff_coeff_second = np.abs(self.current_fit[1] - other_line.current_fit[1])

        return diff_coeff_first < threshold[0] and diff_coeff_second < threshold[1]
       

    def distance_between_best_fit(self, other_line):
        return np.abs(self.best_fit_poly(IMAGE_MAX_Y) - other_line.best_fit_poly(IMAGE_MAX_Y))

    def distance_between_lines(self, other_line):
        return np.abs(self.current_fit_poly(IMAGE_MAX_Y) - other_line.current_fit_poly(IMAGE_MAX_Y))

    def calc_curvature_radius(self, fit_cr):
        y = np.array(np.linspace(0, IMAGE_MAX_Y, num=10))
        x = np.array([fit_cr(x) for x in y])
        y_eval = np.max(y)

        fit_cr = np.polyfit(y * YM_PER_PIXEL, x * XM_PER_PIXEL, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curverad
