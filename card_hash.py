import requests
import numpy as np
from io import BytesIO
from PIL import Image
from picamera import PiCamera
import time
import cv2
import json
import os.path

hashSize = 32
def hashGen():
  global hashSize
  stream = BytesIO()
  camera = PiCamera()

  camera.start_preview()
  time.sleep(2)
  camera.capture(stream, format='jpeg')
  # Construct a numpy array from the stream
  data = np.fromstring(stream.getvalue(), dtype=np.uint8)
  # "Decode" the image from the array, preserving colour
  image = cv2.imdecode(data, 1)
  image = cv2.flip(image, 0)
  cv2.imshow('image',image)
  cv2.waitKey(0)
  image = image[48:227,318:660]
  cv2.imshow('image',image)
  cv2.waitKey(0)
  resized = cv2.resize(image, (hashSize + 1, hashSize))
  diff = resized[:, 1:] > resized[:, :-1]
  imageHash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
  imageHash = format(imageHash, '#0'+str(hashSize**2+2)+'b')
  #print(imageHash)
  return imageHash

# Opening JSON file


cHashes = {}

cHashes[1] = hashGen()
    
jsonString = json.dumps(cHashes)
jsonFile = open("blank.json", "w")
jsonFile.write(jsonString)
jsonFile.close()
