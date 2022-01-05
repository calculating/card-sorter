import io
import time
from picamera import PiCamera
import cv2
import numpy as np
from fractions import Fraction


# function to display the coordinates of
# of the points clicked on the image 
def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  


# Create the in-memory stream
stream = io.BytesIO()
camera = PiCamera(resolution=(820,616))

camera.start_preview()
time.sleep(2)
camera.capture(stream, format='jpeg')
# Construct a numpy array from the stream
data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
# "Decode" the image from the array, preserving colour
imge = cv2.imdecode(data, 1)
imge = cv2.rotate(imge, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('image',imge)
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
img = imge[100:130,130:212]
cv2.imshow('image',img)
cv2.waitKey(0)
img = imge[88:122,493:585]
cv2.imshow('image',img)
cv2.waitKey(0)
img = imge[52:118,283:388]
cv2.imshow('image',img)
cv2.waitKey(0)
img = imge[566:631,202:290]
cv2.imshow('image',img)
cv2.waitKey(0)
img = imge[577:649,359:463]
cv2.imshow('image',img)
cv2.waitKey(0)


