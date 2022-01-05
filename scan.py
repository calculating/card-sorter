import io
import time
from picamera import PiCamera
import cv2
import numpy as np

stream = io.BytesIO()
camera = PiCamera(resolution=(820,616))
camera.start_preview()
time.sleep(2)

def scan():
	camera.capture(stream, format='jpeg')
	# Construct a numpy array from the stream
	data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
	# "Decode" the image from the array, preserving colour
	sideways = cv2.imdecode(data, 1)
	img = cv2.rotate(sideways, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	crop = img[3:29,58:145]
	lineify(crop)
	crop = img[2:22,496:590]
	lineify(crop)
	crop = img[603:661,412:467]
	lineify(crop)
	crop = img[590:649,203:249]
	lineify(crop)


def lineify(inputImage):
	cv2.imshow('image',inputImage)
	cv2.waitKey(0)
	grey = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
	kernel_size = 15
	grey = cv2.GaussianBlur(grey,(kernel_size, kernel_size),0)
	#(thresh, im_bw) = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	im_bw = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
	cv2.imshow('image',im_bw)
	cv2.waitKey(0)
	
	low_threshold = 10
	high_threshold = 250
	im_bw = cv2.Canny(im_bw, low_threshold, high_threshold)
	
	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 180  # angular resolution in radians of the Hough grid
	threshold = 1  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 5  # minimum number of pixels making up a line
	max_line_gap = 4  # maximum gap in pixels between connectable line segments
	
	
	cv2.imshow('image',im_bw)
	cv2.waitKey(0)
	
	im_bw = cv2.resize(im_bw, (0,0), fx=2, fy=2)
	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(im_bw, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)

	#for line in lines:
	#	for x1,y1,x2,y2 in line:
	#		cv2.line(inputImage,(x1,y1),(x2,y2),(255,0,0),1)
	#print(lines)
	x = int((lines[0][0][1]+lines[0][0][3])/4)
	y = int((lines[0][0][0]+lines[0][0][2])/4)
	inputImage[x, y] = [0, 255, 0]
	cv2.imshow('image',inputImage)
	cv2.waitKey(0)

scan()
