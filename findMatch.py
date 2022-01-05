import requests
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import json
import imutils
import colorsys

# Opening JSON file
data = json.load(open('d_hash_64bit_crop.json',encoding="utf8"))
data2 = json.load(open('d_hash_256bit_crop.json',encoding="utf8"))
data3 = json.load(open('hues.json',encoding="utf8"))


def dhash(image, hashSize=8):
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	diff = resized[:, 1:] > resized[:, :-1]
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
def hashGen(image):
  global prevStr
  image = image[int(image.shape[0]/8):int(image.shape[0]/2),int(image.shape[1]/10):int(image.shape[1]*(9/10))]
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow("e", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  imageHash = dhash(image)
  imageHash = format(imageHash, '#066b')
  return imageHash


def dhash2(image, hashSize=16):
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    resized2 = cv2.resize(resized, ((hashSize + 1)*50, hashSize*50))
    cv2.imshow("small", resized2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
def hashGen2(image):
  global prevStr
  #image = image = image[0:int(image.shape[0]/2)]
  #image = image[int(image.shape[0]/100):int(image.shape[0]/2),int(image.shape[1]/100):int(image.shape[1]*(99/100))]
  image = image[int(image.shape[0]/8):int(image.shape[0]/2),int(image.shape[1]/10):int(image.shape[1]*(9/10))]
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow("e", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  imageHash = dhash2(image)
  imageHash = format(imageHash, '#0258b')
  print(imageHash)
  return imageHash

def hueGen(image):
  global prevStr
  image = image[int(image.shape[0]/8):int(image.shape[0]/2),int(image.shape[1]/10):int(image.shape[1]*(9/10))]
  avg_color_per_row = np.average(image, axis=0)
  avg_color = np.average(avg_color_per_row, axis=0)
  avg_color = colorsys.rgb_to_hls(avg_color[0], avg_color[1], avg_color[2])[0]
  return avg_color

def hamming(a, b):
    return bin(a^b).count('1')



def hashMatch(dHash, hashList):
    hamDict = {}
    card = "id"
    for i in hashList:
        ham = hamming(int(dHash, base=2), int(hashList[i], base=2))
        hamDict[i] = ham
    #sortedHam = sorted(hamDict.items(), key=lambda x: x[1])
    #hamDict = sortedHam[:10]
    return hamDict
        

def hashMatchOld(dHash, hashList):
    smallHam = 999999
    card = "id"
    for i in hashList:
        ham = hamming(int(dHash, base=2), int(hashList[i], base=2))
        if ham < smallHam:
            smallHam = ham
            #print(smallHam)
            card = i
    print(card)
    res = (requests.get('https://api.scryfall.com/cards/'+card).json())
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

    

def scan(url):
    img = Image.open(BytesIO(requests.get(url).content))
    image = np.array(img)
    image = imutils.resize(image, width=500)
    cv2.imshow("Input", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (51, 51), 0)
    etc,gray = cv2.threshold(gray,155,255,cv2.THRESH_BINARY)
    cv2.imshow("Binary", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    edged = cv2.Canny(gray, 75, 200)
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    oldPts = np.float32(screenCnt)
    newPts = np.float32([[0, 0], [0, 204], [146, 204], [146, 0]])
    #print(oldPts)
    #print(newPts)
    matrix = cv2.getPerspectiveTransform(oldPts, newPts)
    result = cv2.warpPerspective(image, matrix, (146, 204))
    cv2.imshow("Scan", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result

def finalize(book1, book2):
    top = dict()

    for guess in book1:
        top[guess] = book1[guess] + book2[guess]

    sortedTop = sorted(top.items(), key=lambda x: x[1])
    top = sortedTop[:10]
    #print(top)
    #first = top[0][0]
    book1 = sorted(book1.items(), key=lambda x: x[1])
    print(book1[0][0])
    #first = book1[0][0]
    book2 = sorted(book2.items(), key=lambda x: x[1])
    print(book2[0][0])
    first = book2[0][0]
    card = str(first)
    print(card)
    res = (requests.get('https://api.scryfall.com/cards/'+card).json())
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

def colorConfirm(book, hue):
    global data3
    #print(book)
    closest = ""
    smallNum = 9.9
    for guess in book:
        #print(data3[guess[0]])
        if min(abs(data3[guess[0]]-hue), 1-abs(data3[guess[0]]-hue)) < smallNum:
            smallNum = min(abs(data3[guess[0]]-hue), 1-abs(data3[guess[0]]-hue))
            closest = guess[0]

    card = closest
    print(card)
    res = (requests.get('https://api.scryfall.com/cards/'+card).json())
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
    


def search(url):
    scannedCard = scan(url)
    hash256 = hashGen2(scannedCard)
    hashMatchOld(hash256, data2)
    #hash64crop = hashGen(scannedCard)
    #hash256 = hashGen2(scannedCard)
    #hue = hueGen(scannedCard)
    #colorConfirm(hashMatch(hash256, data2), hue)
    #dict64 = hashMatch(hash64crop, data)
    #dict256 = hashMatch(hash256, data2)
    #finalize(dict64, dict256)

badPics = [
    "https://i.ibb.co/sw9xzJZ/PXL-20210715-020302754.jpg", 
    "https://i.ibb.co/898RRv1/PXL-20210715-020040425.jpg",
    "https://i.ibb.co/VY88M6P/PXL-20210715-020008654.jpg"
    ]

realPics = [
    "https://c1.scryfall.com/file/scryfall-cards/small/front/2/8/281a685a-bd02-43bf-8700-2207c65bbbb1.jpg?1562904521", 
    "https://c1.scryfall.com/file/scryfall-cards/small/front/6/1/614b9df9-c959-4bdb-91c0-75ae60b724e4.jpg?1567754665",
    "https://c1.scryfall.com/file/scryfall-cards/small/front/d/d/dd435013-0ab9-42f4-985c-66ea2b3760e9.jpg?1562478097", 
    "https://c1.scryfall.com/file/scryfall-cards/small/front/2/f/2f752339-003d-4ded-b2bf-e4200fc8d5d6.jpg?1562903760",
    "https://c1.scryfall.com/file/scryfall-cards/small/front/d/9/d9d2bfa3-0499-43ea-a76d-b12fddbc104e.jpg?1562935702",
    "https://c1.scryfall.com/file/scryfall-cards/small/front/0/d/0daa5458-2a97-40d0-b18d-2381a7a68ee1.jpg?1562897807"
    ]

#for link in realPics:
#    hashMatchOld(hashGen2(np.array(Image.open(BytesIO(requests.get(link).content)))), data2)
 
pictures = [
    "https://i.ibb.co/J2PVBNM/PXL-20210714-191001887.jpg",
    "https://i.ibb.co/47KVbxh/PXL-20210714-190946423.jpg",
    "https://i.ibb.co/Pwf3bLX/PXL-20210714-190938195.jpg",
    "https://i.ibb.co/Zc8sJVD/PXL-20210714-190932095.jpg",
    "https://i.ibb.co/hZBMG7f/PXL-20210714-190925340.jpg",
    "https://i.ibb.co/xhJQ6mh/PXL-20210714-190915341.jpg",
    "https://i.ibb.co/vLGhpqS/PXL-20210714-190901771.jpg",
    "https://i.ibb.co/hcZb33w/PXL-20210715-020142811.jpg",
    "https://i.ibb.co/Y0rn27G/PXL-20210715-020254435.jpg",
    "https://i.ibb.co/ynNL46c/PXL-20210715-015947836.jpg",
    "https://i.ibb.co/LrvJgfj/PXL-20210715-015958713.jpg",
    "https://i.ibb.co/VY88M6P/PXL-20210715-020008654.jpg",
    "https://i.ibb.co/ws9c8CZ/PXL-20210715-020016234.jpg",
    "https://i.ibb.co/PzVzZcC/PXL-20210715-020029551.jpg",
    "https://i.ibb.co/898RRv1/PXL-20210715-020040425.jpg",
    "https://i.ibb.co/Lz1ZH7C/PXL-20210715-020052303.jpg",
    "https://i.ibb.co/rw65LVK/PXL-20210715-020103780.jpg",
    "https://i.ibb.co/sw9xzJZ/PXL-20210715-020302754.jpg"]

#hashMatch(hashGen(np.array(Image.open(BytesIO(requests.get("https://c1.scryfall.com/file/scryfall-cards/small/front/3/6/366ac097-39cb-4401-87c7-1dd73bf34329.jpg?1591228722" ).content)))), data)
for link in pictures:
    search(link)
    #hashMatch(hashGen(scan(link)), data)

