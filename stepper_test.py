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
from sympy import Point, Segment
from sympy import Line as ray
import sys
from statistics import mean
from collections import Counter

debug = False
visual = False

ramp = Motor(6, 5)
disp = Motor(27, 17)

stepD = OutputDevice(23) # CW+
stepE = OutputDevice(24) # CLK+
pos=1
#blank = hashGen(cap()[601:620,123:196], 32)
print("moving to position ", pos)
steps = 400*(pos-4)
if (steps < 0):
    stepD.off()
else:
    stepD.on()
steps = abs(steps)

if (pos == 7 or pos == 1):
    steps -= 200

for x in range(steps):
    stepE.toggle()
    sleep(0.002)

bd()
stepD.toggle()

for x in range(steps):
    stepE.toggle()
    sleep(0.002)
    #print(hamming(int(hashGen(cap()[601:620,123:196], 32), base=2), int(blank, base=2)))
sleep(.001)