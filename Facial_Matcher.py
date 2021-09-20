import cv2
import numpy as np
import imutils

main_path = '/Users/dionisios/src/face_matcher/'

imgRaw = cv2.imread(main_path+"Resources/Straight_norm.jpg")
template = cv2.imread(main_path+"Resources/Picture.jpg")
head_out = cv2.imread(main_path+"Resources/head_outline.jpg")

# Percent by which the image is resized
scale_percent = 20
# Calculate the percentage of original dimensions
width = int(imgRaw.shape[1] * scale_percent / 100)
height = int(imgRaw.shape[0] * scale_percent / 100)
# Adjust image
dsize = (width, height)
imgAdj = cv2.resize(imgRaw, dsize)

# Convert to greyscale, blur, apply Canny edge detection and invert colour
imgGrey = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
imgCanny = cv2.Canny(imgBlur, 50, 150)
invCanny = cv2.bitwise_not(imgCanny)

# Flip the template image horizontally (left-to-right)
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

# Call the Haar cascade classifier with a pre-trainer profile face model
face_cascade = cv2.CascadeClassifier("Resources/haarcascade_profileface.xml")

# Apply face detection with the Haar cascade classifier
faces = face_cascade.detectMultiScale(imgGrey, 1.1, 5)

# For each face detected, draw a rectangle around it and get the Region Of Interest (ROI) from the inverse Canny
for (x, y, w, h) in faces:
    cv2.rectangle(imgGrey, (x, y), (x + w, y + h), (255, 255, 255), 2)
    print(x, y, w, h)
    roi = invCanny[y - 200:y + (h + 200), x:x + (w + 300)]


def alignImages(image, template, debug=True):
    """Matches the keypoints of the image and template

    Args:
        image ([np.array]): Main image
        template (np.array): Template
        debug (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    image = image
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Make named windows for our ROI and template and set mouse call backs for adding points
    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", getImageKpts)

    cv2.namedWindow("Template")
    cv2.setMouseCallback("Template", getTemplateKpts)

    imgpts = []
    tempts = []

    # Begin while loop awaiting mouse input on windows
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

    # Turn points into keypoints
    pts1 = [cv2.KeyPoint(imgpts[0][0], imgpts[0][1], 1), cv2.KeyPoint(imgpts[1][0], imgpts[1][1], 1), cv2.KeyPoint(imgpts[2][0], imgpts[2][1], 1),
             cv2.KeyPoint(imgpts[3][0], imgpts[3][1], 1), cv2.KeyPoint(imgpts[4][0], imgpts[4][1], 1), cv2.KeyPoint(imgpts[5][0], imgpts[5][1], 1)]

    pts2 = [cv2.KeyPoint(tempts[0][0], tempts[0][1], 1), cv2.KeyPoint(tempts[1][0], tempts[1][1], 1), cv2.KeyPoint(tempts[2][0], tempts[2][1], 1),
             cv2.KeyPoint(tempts[3][0], tempts[3][1], 1), cv2.KeyPoint(tempts[4][0], tempts[4][1], 1), cv2.KeyPoint(tempts[5][0], tempts[5][1], 1)]

    # Create ORB object and compute descriptors of the keypoints
    orb = cv2.ORB_create(6)
    (kpts1, des1) = orb.compute(image, pts1)
    (kpts2, des2) = orb.compute(template, pts2)

    # Create a Brute-Force Matcher, to match keypoints between image and template
    matcher = cv2.BFMatcher()
    matches = matcher.match(des1, des2, None)
    matches = sorted(matches, key=lambda x: x.distance)

    if debug:
        matchedVis = cv2.drawMatches(image, kpts1, template, kpts2, matches, outImg=None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.imwrite("Aligned.jpg", matchedVis)
        cv2.waitKey(0)

    # # Transformation matrix, T
    # T_s = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])  # Scaling matrix
    # T_r = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # Rotational matrix
    # T = T_s @ T_r
    #
    # Transform = np.float32(T.flatten()[:6].reshape(2, 3))
    # img_transformed = cv2.warpAffine(image, Transform, (100, 100))
    # cv2.imshow("transformed", img_transformed)
    # cv2.waitKey(0)
    #
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    list_kpts1 = []
    list_kpts2 = []

    # loop over the top matches
    for m in matches:

        # Get the matching keypoints for each of the images
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kpts1[img1_idx].pt
        (x2, y2) = kpts2[img2_idx].pt

        # Append to each list
        list_kpts1.append((x1, y1))
        list_kpts2.append((x2, y2))

        list_kpts1 = [kpts1[m.queryIdx].pt for m in matches]
        list_kpts2 = [kpts2[m.trainIdx].pt for m in matches]


    # compute the homography matrix between the two sets of matched
    # points
    (H, mask) = cv2.findHomography(list_kpts1, list_kpts2, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    # return the aligned image
    return aligned

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

# Our main image is now the ROI inverse Canny image, while the main tepmlate is the inverse template
image = roi
template = templateFlip



align = alignImages(image, template, debug=True)
#
# image_alg = imutils.resize(align, width=500)
# template_alg = imutils.resize(template, width=500)
#
# overlay = template_alg
# output = image_alg
#
# cv2.imshow("overlay", align)
# cv2.waitKey(0)
