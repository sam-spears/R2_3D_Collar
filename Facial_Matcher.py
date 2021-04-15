import cv2
import numpy as np
import imutils

imgRaw = cv2.imread("Resources/straight_norm.jpg")
template = cv2.imread("Resources/Picture.jpg")
head_out = cv2.imread("Resources/head_outline.jpg")

# percent by which the image is resized
scale_percent = 30
# calculate the percentage of original dimensions
width = int(imgRaw.shape[1] * scale_percent / 100)
height = int(imgRaw.shape[0] * scale_percent / 100)
# adjust image
dsize = (width, height)
imgAdj = cv2.resize(imgRaw, dsize)

imgGrey = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
imgCanny = cv2.Canny(imgBlur, 150, 150)
invCanny = cv2.bitwise_not(imgCanny)

templateFlip = cv2.flip(template, 1)

counter = 0
imgRefPt = np.zeros((6, 2), int)

counter1 = 0
temRefPt = np.zeros((6, 2), int)

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getImageKpts(event, x, y, flags, params):
    global counter

    if event == cv2.EVENT_LBUTTONDOWN:
        imgRefPt[counter] = x, y
        counter = counter + 1
        print(imgRefPt)

def getTemplateKpts(event, x, y, flags, params):
    global counter1

    if event == cv2.EVENT_LBUTTONDOWN:
        temRefPt[counter1] = x, y
        counter1 = counter1 + 1
        print(temRefPt)

face_cascade = cv2.CascadeClassifier("Resources/haarcascade_profileface.xml")

faces = face_cascade.detectMultiScale(imgRaw, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(invCanny, (x, y), (x + w, y + h), (255, 255, 255), 2)
    print(x, y, w, h)
    roi = imgGrey[y - 200:y + (h + 200), x:x + (w + 300)]

def alignImages(image, template, MaxFeatures=50, Good_Match_Perc=0.95, debug=True):

    image = image
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", getImageKpts)

    cv2.namedWindow("Template")
    cv2.setMouseCallback("Template", getTemplateKpts)

    imgpts = []
    tempts = []

    while True:

        if counter == 6:
            imgpts = np.float32([imgRefPt[0], imgRefPt[1], imgRefPt[2], imgRefPt[3], imgRefPt[4], imgRefPt[5]])
            tempts = np.float32([temRefPt[0], temRefPt[1], temRefPt[2], temRefPt[3], temRefPt[4], temRefPt[5]])

        for x in range(0, 6):
            cv2.circle(image, (imgRefPt[x][0], imgRefPt[x][1]), 3, (0, 255, 0), cv2.FILLED)
            cv2.circle(template, (temRefPt[x][0], temRefPt[x][1]), 3, (0, 255, 0), cv2.FILLED)

        cv2.imshow("ROI", image)
        cv2.imshow("Template", template)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break

    #turn points into keypoints

    kpts1 = [cv2.KeyPoint(imgpts[0][0], imgpts[0][1], 1), cv2.KeyPoint(imgpts[1][0], imgpts[1][1], 1), cv2.KeyPoint(imgpts[2][0], imgpts[2][1], 1),
             cv2.KeyPoint(imgpts[3][0], imgpts[3][1], 1), cv2.KeyPoint(imgpts[4][0], imgpts[4][1], 1), cv2.KeyPoint(imgpts[5][0], imgpts[5][1], 1)]

    kpts2 = [cv2.KeyPoint(tempts[0][0], tempts[0][1], 1), cv2.KeyPoint(tempts[1][0], tempts[1][1], 1), cv2.KeyPoint(tempts[2][0], tempts[2][1], 1),
             cv2.KeyPoint(tempts[3][0], tempts[3][1], 1), cv2.KeyPoint(tempts[4][0], tempts[4][1], 1), cv2.KeyPoint(tempts[5][0], tempts[5][1], 1)]

    orb = cv2.ORB_create(6)
    (keypoints1, descriptors1) = orb.compute(image, kpts1)
    (keypoints2, descriptors2) = orb.compute(template, kpts2)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = sorted(matches, key=lambda x: x.distance)

    numGoodMatches = int(len(matches) * Good_Match_Perc)
    matches = matches[:numGoodMatches]

    if debug:
        matchedVis = cv2.drawMatches(image, keypoints1, template, keypoints2,
                                     matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.imwrite("Aligned.jpg", matchedVis)
        cv2.waitKey(0)

    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for i, m in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = keypoints1[m.queryIdx].pt
        ptsB[i] = keypoints2[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # return the aligned image
    return aligned


image = roi.copy()
template = templateFlip

align = alignImages(image, templateFlip, debug=True)

image_alg = imutils.resize(align, width=500)
template_alg = imutils.resize(template, width=500)

overlay = template_alg
output = image_alg

cv2.imshow("overlay", align)
cv2.waitKey(0)
