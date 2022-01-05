from gpiozero import Motor, Servo, OutputDevice
from time import sleep
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import json
import imutils
import colorsys
from picamera import PiCamera

debug = "visual"
# console/visual/off

dispenser = Motor(27, 17)
conveyor = Motor(6, 5)

rampDirection = OutputDevice(23) # CW+
rampMicrostep = OutputDevice(24) # CLK+ 400 something

def captureImage():
	stream = BytesIO()
	camera = PiCamera(resolution=(820,616))
	camera.start_preview()
	camera.capture(stream, format='jpeg')
	ImData = np.frombuffer(stream.getvalue(), dtype=np.uint8)
	img = cv2.rotate(cv2.imdecode(ImData, 1), cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
	camera.stop_preview()
	camera.close()
	return img

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def dispenseACard():
	reference = captureImage()[400:820]
	
	if debug == "visual":
		cv2.imshow('ref',reference)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	sleep(1)
	referenceDifference = 2*mse(reference, captureImage()[400:820])
	
	if debug != "off":
		print(referenceDifference)
	
	for intervals in range(20):
		newPicture = captureImage()[400:820]
		
		if debug == "visual":
			cv2.imshow('new',newPicture)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		
		difference = mse(reference, newPicture)
		if debug != "off":
			print(difference)
		if (difference < referenceDifference):
			dispenser.forward(1)
			sleep(.5)
			dispenser.stop()
		else:
			break
	
cv2.imshow('cap',captureImage())
cv2.waitKey(0)
cv2.destroyAllWindows()
