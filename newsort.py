from gpiozero import Motor, OutputDevice
from time import sleep
import numpy as np
from io import BytesIO
import cv2
from picamera import PiCamera

import requests


from PIL import Image

import json
import imutils
import colorsys

from sympy import Point, Segment
from sympy import Line as ray
import sys
from statistics import mean
from collections import Counter


debug = "console"
# console/visual/off

dispenser = Motor(27, 17)
conveyor = Motor(6, 5)

rampDirection = OutputDevice(23) # CW+
rampMicrostep = OutputDevice(24) # CLK+ 400 something


blu = 90
BLUE_MIN = np.array([blu-20, 130, 70], np.uint8)
BLUE_MAX = np.array([blu+20, 255, 255], np.uint8)
#print(BLUE_MAX, BLUE_MIN)

data = json.load(open('hashes_'+str(32**2)+'.json',encoding="utf8"))


def hashMatch(dHash, hashList):
    hamDict = {}
    card = "id"
    for i in hashList:
        ham = hamming(int(dHash, base=2), int(hashList[i], base=2))
        hamDict[i] = ham
    sortedHam = sorted(hamDict.items(), key=lambda x: x[1])
    hamDict = sortedHam[:10]
    #print(hamDict)
    card = hamDict[0][0]
    #cardInfo(card)
    return card, hamDict[0][1]

def hamming(a, b):
    return bin(a^b).count('1')

def artCrop(image):
    image = image[int(image.shape[0]/6):int(image.shape[0]/2),int(image.shape[1]/8):int(image.shape[1]*(7/8))]
    cv2.imshow('new',image)
    cv2.waitKey(0)
    
    return image


def cardInfo(card):
    res = (requests.get('https://api.scryfall.com/cards/'+card).json())
    print(res["name"])
    return(res["cmc"])

def hashGen(image, hashSize = 32):
    #cv2.imshow("hash request", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    image = cv2.GaussianBlur(image,(kernel_size, kernel_size),0)
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    diff = resized[:, 1:] > resized[:, :-1]
    imageHash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    imageHash = format(imageHash, '#0'+str(hashSize**2+2)+'b')
    return imageHash


#image = artCrop(scan(cap()))
#			hashResult = hashMatch(hashGen(image), data)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def captureImage():
    stream = BytesIO()
    camera = PiCamera(resolution=(820,616))
    camera.start_preview()
    camera.capture(stream, format='jpeg')
    ImData = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(ImData, 1)
    camera.stop_preview()
    camera.close()
    return adjust_gamma(img, 2)

def startup():
    # Clear ramp
    # Zero stepper
    return

def dispense():
    reference = captureImage()[70:510,0:450]
    
    if debug == "visual":
        cv2.imshow('ref', reference)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    sleep(1)
    referenceDifference = mse(reference, adjust_gamma(reference, 0.8))
    
    if debug != "off":
        print(referenceDifference)
    
    dispenser.forward(1)
    
    while True:
        newPicture = captureImage()[70:510,0:450]
        
        if debug == "visual":
            cv2.imshow('new', newPicture)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        difference = mse(reference, newPicture)

        if debug != "off":
            print(difference)

        if (difference > referenceDifference):
            dispenser.stop()
            break
    
    sleep(1)

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def scan():
    img = cv2.rotate(captureImage(), cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('new', img)
    cv2.waitKey(0)
    dupe = img[0:800,70:510]
    img = img[0:300,70:510]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    kernel_size = 11
    hsv = cv2.GaussianBlur(hsv,(kernel_size, kernel_size),0)
    threshed = cv2.inRange(hsv, BLUE_MIN, BLUE_MAX)
    cv2.imshow('new', threshed)
    cv2.waitKey(0)
    
    threshed = cv2.bitwise_not(threshed)
    threshed = cv2.morphologyEx(threshed,cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))

    cnts = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    card = max(cnts, key=cv2.contourArea)

    peri = cv2.arcLength(card, True)

    card = cv2.approxPolyDP(card, 0.05 * peri, True)
    
    cv2.drawContours(img, [card], -1, (0, 0, 255), 3)
    print("hmm")
    print(card)
    dist = (((card[0][0][0]-card[3][0][0])**2+(card[0][0][1]-card[3][0][1])**2)**(0.5))*1.4
    distRight =  (((card[2][0][0]-card[3][0][0])**2+(card[2][0][1]-card[3][0][1])**2)**(0.5))
    ratioX = (card[3][0][0]-card[2][0][0])/distRight
    ratioY = (card[3][0][1]-card[2][0][1])/distRight
    card[1][0] = [card[0][0][0]-dist*ratioX, card[0][0][1]-dist*ratioY]
    card[2][0] = [card[3][0][0]-dist*ratioX, card[3][0][1]-dist*ratioY]
    print("hmmm")
    print(card)
    
    fullImg = dupe
    #cv2.drawContours(fullImg, [card], -1, (0, 0, 255), 3)
    #print(card[0][0][0])
    #print(card[0][0][1])
    #print(card[1][0][1])
    #distTop = card[0][0][0]
    
    
    cv2.imshow('new', img)
    cv2.waitKey(0)
    cv2.imshow('new', fullImg)
    cv2.waitKey(0)
    
    oldPts = np.float32([card[0][0], card[3][0], card[1][0], card[2][0]])
    newPts = np.float32([[0, 0], [0, 146], [204, 0], [204, 146]])
    matrix = cv2.getPerspectiveTransform(oldPts, newPts)
    result = cv2.warpPerspective(fullImg, matrix, (204, 146))
    result = cv2.rotate(result, cv2.cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('new', result)
    cv2.waitKey(0)
    
    hashResult = hashMatch(hashGen(artCrop(result)), data)[0]
    print(hashResult)
    #cardInfo(hashResult)
    
    return

def sort(card):
    return

def deposit(bin):
    return

startup()

while True:
    #dispense()

    card = scan()

    bin = sort(card)

    deposit(bin)
