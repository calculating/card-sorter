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

data = json.load(open('hashes_'+str(32**2)+'.json',encoding="utf8"))


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

def intercepts(line):
	x1 = line[0]
	y1 = line[1]
	x2 = line[2]
	y2 = line[3]
	a = (y2 - y1) / (x2 - x1)
	b = y1 - a * x1  
	b2 = a*616 +b   
	return b, b2

def scan(fullImg, flipped = False):
	#cv2.imshow('image',fullImg)
	#cv2.waitKey(0)
	cropMtrx = [
	[130,169,227,293, 'v', [0, -1], 'B'], 
	[437,176,525,291, 'v', [0, 1], 'B'], 
	[201,68,290,159, 'h', [1, -1], 'B'],
	[371,60,446,164, 'h', [1, -1], 'B'],
	[215,501,309,600, 'h', [1, 1], 'R'],
	[370,507,464,601, 'h', [1, 1], 'R']]
	
	greatLines = []
	
	for bounds in cropMtrx:
		cImg = fullImg[bounds[2]:bounds[3],bounds[0]:bounds[1]]
		
		cv2.imshow('image1'+str(bounds[0]),cImg)
		cv2.imshow("im", fullImg)
		cv2.waitKey(0)
		
		if debug:
			cv2.imshow('image1'+str(bounds[0]),cImg)
			cv2.waitKey(0)
		(B, G, R) = cv2.split(cImg)
		
		if (bounds[6] == 'B'):
			gray = B
		else:
			gray = R
		
		#gray = cv2.cvtColor(cImg,cv2.COLOR_BGR2GRAY)
		
		norm_img = np.zeros((820,616))
		gray = cv2.normalize(gray,  norm_img, 0, 255, cv2.NORM_MINMAX)
		
		cImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
		
		#gray = gray*2
		
		#gray = cv2.cvtColor(cImg, cv2.COLOR_BGR2GRAY)
		k = 7
		blurred = cv2.GaussianBlur(gray, (k, k), 0)
		#cv2.imwrite("zample_"+str(t)+"_2.jpg", blurred)
		canny = cv2.Canny(blurred, 100, 170)
		expansion = 3
		kernel = np.ones((expansion,expansion),np.uint8)
		canny = cv2.dilate(canny,kernel,iterations = 1)
		#k = 3
		#canny = cv2.GaussianBlur(canny, (k, k), 0)
		if debug:
			cv2.imshow('image2'+str(bounds[0]),canny)
			cv2.waitKey(30)
		
		maxDist = int((bounds[1]-bounds[0])*0.1)
		minLen = int((bounds[1]-bounds[0])*0.5)
		lines = cv2.HoughLinesP(canny, rho = 1, theta = 1*np.pi/180, threshold = 10, minLineLength = minLen, maxLineGap = maxDist)
		
		limage = cImg.copy()
		for line in lines:
			limage = cv2.line(limage, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 1)
		if debug:
			cv2.imshow('image3'+str(bounds[0]),limage)
			cv2.waitKey(30)
		
		goodLines = []
		
		threshSlope = int((bounds[1]-bounds[0])*0.3)
		
		for line in lines:
			if (bounds[4] == 'v'):
				if ( abs(line[0][0]-line[0][2]) < threshSlope):
					goodLines.append(line)
			else:
				if ( abs(line[0][1]-line[0][3]) < threshSlope):
					goodLines.append(line)
		
		limage = cImg.copy()
		for line in goodLines:
			limage = cv2.line(limage, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 2)
		#cv2.imshow('image',limage)
		#cv2.waitKey(0)
		
		est = "est"
		
		for line in goodLines:
			#print(line[0][bounds[5][0]]*bounds[5][1])
			if (est == "est"):
				est = [line[0][bounds[5][0]]*bounds[5][1], line[0]]
			if (est[0] < line[0][bounds[5][0]]*bounds[5][1]):
				est = [line[0][bounds[5][0]]*bounds[5][1], line[0]]
				
		limage = cImg.copy()
		#limage = cv2.line(image, start_point, end_point, color, thickness)
		limage = cv2.line(limage, (est[1][0], est[1][1]), (est[1][2], est[1][3]), (0, 255, 0), 2)
		
		
		est[1][0] += bounds[2]
		est[1][2] += bounds[2]
		est[1][1] += bounds[0]
		est[1][3] += bounds[0]
		
		greatLines.append(est[1])
		
		#if debug:
		#	cv2.imshow('image4'+str(bounds[0]),limage)
		#	cv2.waitKey(30)
		
	#for line in greatLines:
	#	fullImg = cv2.line(fullImg, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3)
	

	
	rL = greatLines[2]
	lL = greatLines[3]
	greatLines[2] = [mean([rL[0], rL[2]]), mean([rL[1], rL[3]]), mean([lL[0], lL[2]]), mean([lL[1], lL[3]])]
	
	rL = greatLines[4]
	lL = greatLines[5]
	greatLines[4] = [mean([rL[0], rL[2]]), mean([rL[1], rL[3]]), mean([lL[0], lL[2]]), mean([lL[1], lL[3]])]
	
	
	topL = intercepts(greatLines[2])
	bottomL = intercepts(greatLines[4])
	
	#print(topL, bottomL)
	#cv2.imshow('image',fullImg)
	#cv2.waitKey(0)
	
	
	oldPts = np.float32([[0, topL[0]], [0, bottomL[0]], [616, topL[1]], [616, bottomL[1]]])
	newPts = np.float32([[0, 0], [0, 204], [616, 0], [616, 204]])
	matrix = cv2.getPerspectiveTransform(oldPts, newPts)
	result = cv2.warpPerspective(fullImg, matrix, (616, 204))
	
	#print(greatLines[0][0],greatLines[1][0])
	oldPts = np.float32([[int(greatLines[0][0])+10, 0], [int(greatLines[0][0])+10, 204], [int(greatLines[1][0]), 0], [int(greatLines[1][0]), 204]])
	newPts = np.float32([[0, 0], [0, 204], [146, 0], [146, 204]])
	matrix = cv2.getPerspectiveTransform(oldPts, newPts)
	result = cv2.warpPerspective(result, matrix, (146, 204))
	#result = result[0:204,int(greatLines[0][0]):int(greatLines[1][0])]
	#result = cv2.resize(result, (146, 204), interpolation = cv2.INTER_AREA)
	
	#cv2.imwrite("zample_"+str(t)+"_2.jpg", result)
	
	if flipped:
		result = cv2.rotate(result, cv2.cv2.ROTATE_180)
	
	if visual:
		cv2.imshow('image',result)
		cv2.waitKey(300)
	
	#image=result
	#image = image[int(image.shape[0]/6):int(image.shape[0]/2),int(image.shape[1]/8):int(image.shape[1]*(7/8))]
  
	
	cv2.destroyAllWindows()
	return result
	
def cardInfo(card):
	res = (requests.get('https://api.scryfall.com/cards/'+card).json())
	print(res["name"])
	#if "image_uris" in res:
	#	result = res["image_uris"]["normal"]
	#else:
	#	if "card_faces" in res:
	#		if "image_uris" in res["card_faces"][0]:
	#			result = res["card_faces"][0]["image_uris"]["normal"]
	#img = Image.open(BytesIO(requests.get(result).content))
	#result = np.array(img)
	#result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	#cv2.imshow("result", result)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return(res["cmc"])
		
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

def cap():
	stream = BytesIO()
	camera = PiCamera(resolution=(820,616))
	camera.start_preview()
	camera.capture(stream, format='jpeg')
	ImData = np.frombuffer(stream.getvalue(), dtype=np.uint8)
	img = cv2.rotate(cv2.imdecode(ImData, 1), cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
	camera.stop_preview()
	camera.close()
	#cv2.imshow('image',img)
	#cv2.waitKey(300)
	return img

def hamming(a, b):
    return bin(a^b).count('1')

def dsp():
	disp.backward(1)
	sleep(1.5)
	disp.stop()
	
	blank = hashGen(cap(), 64)
	
	#capture = cap()[464:611,204:441]
	pic = hashGen(cap(), 64)
	#cv2.imshow('image',capture)
	#cv2.waitKey(300)
	ref = int(hamming(int(pic, base=2), int(blank, base=2))*1.2)
	
	print("reference similarity: ", ref)
	dif = 0
	
	for d in range(20):
		pic = hashGen(cap(), 64)
		dif = hamming(int(pic, base=2), int(blank, base=2)) 
		print(dif)
		if (dif < ref):
			disp.forward(1)
			sleep(.5)
			disp.stop()
		else:
			break

def artCrop(image):
	image = image[int(image.shape[0]/6):int(image.shape[0]/2),int(image.shape[1]/8):int(image.shape[1]*(7/8))]
	return image

def bp(pos):
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

def bd():
	ramp.forward(1)
	sleep(1)
	ramp.stop()



bp(4)
bp(4)
while True:
	dsp()
	cards = []
	sim1 = 9999999
	trys = 0
	for n in range(5):
		try:
			image = artCrop(scan(cap()))
			hashResult = hashMatch(hashGen(image), data)
			cards.append(hashResult[0])
			if hashResult[1] < sim1:
				sim1 = hashResult[1]
		except:
			print("small oop")
			trys += 1
			if trys < 10:
				n -= 1
			
	
	cards2 = []
	sim2 = 9999999
	trys = 0
	for n in range(5):
		try:
			image = artCrop(scan(cap(), True))
			hashResult = hashMatch(hashGen(image), data)
			cards2.append(hashResult[0])
			if hashResult[1] < sim2:
				sim2 = hashResult[1]
		except:
			print("small oop")
			trys += 1
			if trys < 10:
				n -= 1
	
	try:
		cardid1 = Counter(cards).most_common(1)[0][0]
		cardid2 = Counter(cards2).most_common(1)[0][0]
		print(cardid1, cardid2)
		
		if sim1 < sim2:
			cmc = cardInfo(cardid1)
		else:
			cmc = cardInfo(cardid2)
		
		if cmc == 0:
			bp(2)
		elif cmc == 1:
			bp(2)
		elif cmc == 2:
			bp(3)
		elif cmc == 3:
			bp(3)
		elif cmc == 4:
			bp(5)
		elif cmc == 5:
			bp(5)
		elif cmc > 5:
			bp(6)
		else:
			bp(4)
	except Exception as ex:
		print("oop  ", ex)
		bp(4)
		
	

