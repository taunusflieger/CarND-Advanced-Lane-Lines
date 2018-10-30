# Advanced Lane fining main module


from LaneFinder import LaneFinder
from RawImageProcessor import RawImageProcessor
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

TEST_PICTURE_PATH = 'Term1/CarND-Advanced-Lane-Lines/test_images/'
OUTPUT_PICTURE_PATH = 'Term1/CarND-Advanced-Lane-Lines/output_images/'
TEST_VIDEO_PATH = 'Term1/CarND-Advanced-Lane-Lines/'
CAM_CAL_PICTURE_PATH = 'Term1/CarND-Advanced-Lane-Lines/camera_cal/'

# Detect lane lines in an entire video and write the result to disc
def process_video(video_name):
    laneFinder = LaneFinder(n_frames=5)

    video_input = VideoFileClip(TEST_VIDEO_PATH + video_name + ".mp4")
    video_output = TEST_VIDEO_PATH + video_name + "_output.mp4"
    output = video_input.fl_image(laneFinder.process_video_frame)
    output.write_videofile(video_output, audio=False)

def process_image(name):
    image_file_name = TEST_PICTURE_PATH + name + '.jpg'
    output_image_file = OUTPUT_PICTURE_PATH + name + '_output.jpg'
    print(image_file_name)
    laneFinder = LaneFinder()
    image = mpimg.imread(image_file_name) 
    result_image = laneFinder.process_image(image, True)
    mpimg.imsave(output_image_file, result_image)

def process_sample_images():
    for i in range(6):
        image_file_name = TEST_PICTURE_PATH + 'test' + str(i+1) + '.jpg'
        output_image_file = OUTPUT_PICTURE_PATH + 'test' + str(i+1) + '_output.jpg'
        print(image_file_name)
        laneFinder = LaneFinder()
        image = mpimg.imread(image_file_name) 
        result_image = laneFinder.process_image(image, True)
        mpimg.imsave(output_image_file, result_image)

def undistort_image(name):
    image_file_name = TEST_PICTURE_PATH + name + '.jpg'
    output_image_file = OUTPUT_PICTURE_PATH + name + '_undist_output.jpg'

    rawProcessor = RawImageProcessor()
    image = mpimg.imread(image_file_name) 
    result_image = rawProcessor.undistort(image)

    mpimg.imsave(output_image_file, result_image)


if __name__ == "__main__":
    #process_sample_images()
    process_video('project_video')
    #undistort_image('test1')
    print('...')

