import requests
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import json
import imutils
import colorsys
import time
from picamera import PiCamera

hashSize = 32
print('hashes_'+str(hashSize**2)+'.json')
data = json.load(open('hashes_'+str(hashSize**2)+'_blurred.json',encoding="utf8"))
#print(data)

def hashGen(image):
  global hashSize
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  resized = cv2.resize(image, (hashSize + 1, hashSize))
  diff = resized[:, 1:] > resized[:, :-1]
  imageHash = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
  imageHash = format(imageHash, '#0'+str(hashSize**2+2)+'b')
  return imageHash

def hamming(a, b):
    return bin(a^b).count('1')

def hashMatch(dHash, hashList):
    hamDict = {}
    card = "id"
    for i in hashList:
        ham = hamming(int(dHash, base=2), int(hashList[i], base=2))
        hamDict[i] = ham
    sortedHam = sorted(hamDict.items(), key=lambda x: x[1])
    hamDict = sortedHam[:10]
    print(hamDict)
    print(hamming(int(dHash, base=2), int(hashList["6ed0f6a5-ed40-44fc-a5e1-3f8bb968d1d9"], base=2)))
    card = hamDict[0][0]
    resultingCard(card)
    
def resultingCard(cardID):
    res = (requests.get('https://api.scryfall.com/cards/'+cardID).json())
    print(res["name"])
    if "image_uris" in res:
            result = res["image_uris"]["normal"]
    else:
        if "card_faces" in res:
            if "image_uris" in res["card_faces"][0]:
                result = res["card_faces"][0]["image_uris"]["normal"]
    img = Image.open(BytesIO(requests.get(result).content))
    result = np.array(img)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

stream = BytesIO()
camera = PiCamera()

camera.start_preview()
time.sleep(2)
camera.capture(stream, format='jpeg')
ImData = np.fromstring(stream.getvalue(), dtype=np.uint8)
img = cv2.imdecode(ImData, 1)
img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("initial", img)
cv2.waitKey(0)
kernel_size = 51
img = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
cv2.destroyAllWindows()
scannedCard = img[64:293,148:509]
cv2.imshow("scan", scannedCard)
cv2.waitKey(0)
cv2.destroyAllWindows()
hash1028 = hashGen(scannedCard)
hashMatch(hash1028, data)


def search(url):
    scannedCard = scan(url)
    hash1028 = hashGen(scannedCard)
    hashMatch(hash1028, data)

cv2.destroyAllWindows()

