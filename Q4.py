# import OpenCV and pyplot
import cv2 as cv
from matplotlib import pyplot as plt

# read left and right images
imgR = cv.imread('images/IMG-1318.jpg', 0)
imgL = cv.imread('images/IMG-1318_1.jpg', 0)

scale_percent = 60  # percent of original size
width = int(imgR.shape[1] * scale_percent / 100)
height = int(imgR.shape[0] * scale_percent / 100)
dim = (width, height)

imgR = cv.resize(imgR, dim, interpolation=cv.INTER_AREA)
imgL = cv.resize(imgL, dim, interpolation=cv.INTER_AREA)

# creates StereoBm object
stereo = cv.StereoBM_create(numDisparities=16,
                            blockSize=15)

# computes disparity
disparity = stereo.compute(imgL, imgR)

# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()
