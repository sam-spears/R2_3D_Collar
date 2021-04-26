import cv2
import imutils
import math
import numpy as np

image = cv2.imread("Resources/curved_dropped.jpg")
template = cv2.imread("Resources/Picture.jpg")

# percent by which the image is resized
scale_percent = 20
# calculate the percentage of original dimensions
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
# adjust image
dsize = (width, height)
imgAdj = cv2.resize(image, dsize)
img = imgAdj.copy()

# counter = 0
# imgRefPt = np.zeros((6, 2), int)
#
# counter1 = 0
# temRefPt = np.zeros((6, 2), int)
#
# def stackImages(scale, imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
#
# def getImageKpts(event, x, y, flags, params):
#     global counter
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         imgRefPt[counter] = x, y
#         counter = counter + 1
#         print(imgRefPt)
#
# def getTemplateKpts(event, x, y, flags, params):
#     global counter1
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         temRefPt[counter1] = x, y
#         counter1 = counter1 + 1
#         print(temRefPt)
#
# while True:
#
#         if counter == 6:
#             imgpts = np.float32([imgRefPt[0], imgRefPt[1], imgRefPt[2], imgRefPt[3], imgRefPt[4], imgRefPt[5]])
#
#         if counter1 == 6:
#             tempts = np.float32([temRefPt[0], temRefPt[1], temRefPt[2], temRefPt[3], temRefPt[4], temRefPt[5]])
#
#         for x in range(0, 6):
#             cv2.circle(image, (imgRefPt[x][0], imgRefPt[x][1]), 3, (0, 255, 0), cv2.FILLED)
#             cv2.circle(template, (temRefPt[x][0], temRefPt[x][1]), 3, (0, 255, 0), cv2.FILLED)
#
#         cv2.imshow("ROI", template)
#         cv2.setMouseCallback("ROI", getImageKpts)
#         cv2.waitKey(1)

refPt = []
cropping = False
def crop(event, x, y, flags, param):
    global refPt, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        cv2.rectangle(imgAdj, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", imgAdj)

pointsList = []
def mousePoints(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 !=0:
            cv2.line(img,tuple(pointsList[round((size-1)/3)*3]),(x, y), (0,0,255),2)
        cv2.circle(img, (x, y), 5, (0,0,255), cv2.FILLED)
        pointsList.append([x,y])

def gradient(pt1, pt2):
    return (pt2[1]-pt1[1])/(pt2[0]-pt1[0]+1e-10)

def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2-m1)/(1+(m2*m1)))
    angD = round(math.degrees(angR))
    return angD

while True:

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)

    cv2.imshow("Angle Finder", img)
    cv2.setMouseCallback("Angle Finder", mousePoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

clone = imgAdj.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", crop)

while True:

    cv2.imshow("image", imgAdj)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()

    elif key == ord("c"):
        break

if len(refPt) == 2:
    angle = getAngle(pointsList)
    roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    out = imutils.rotate(roi, angle, None)
    cv2.imshow("ROI", out)
    cv2.waitKey(0)

cv2.destroyAllWindows()
