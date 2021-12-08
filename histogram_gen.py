import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import imutils
import argparse
import cv2


def getFrames(video):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    img_list = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = scaleImage(image, 20)

        img_list.append(image)
        success, image = vidcap.read()
        count += 1
    img_array = np.array(img_list)
    return img_array


def temporalGaussianDerivative(frame, sigma_t):
    return frame * (np.exp(-frame / sigma_t) ** 2)


def spatialGaussian(image, sigma_x, sigma_y):

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            image[x, y] = image[x, y] * np.exp(
                -((x / sigma_x) ** 2) + (y / sigma_y) ** 2
            )
    # TODO: need to normalize
    return image


def spatialHistogram(image):
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            pass
    return


def scaleImage(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def frameDeltaGivenPureBackground(video):
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    img_list = []
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = scaleImage(image, 100)

        blurred_img = cv2.GaussianBlur(image, (21, 21), 0)

        img_list.append(image)
        success, image = vidcap.read()
        count += 1

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
    args = vars(ap.parse_args())

    frameDelta = cv2.absdiff(img_list[0], img_list[60])
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it

        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_list[1], (x, y), (x + w, y + h), (0, 0, 0), 2)
        text = "Occupied"

    cv2.imshow("im1", img_list[0])
    cv2.imshow("im2", img_list[60])
    cv2.imshow("Frame Delta", frameDelta)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey()

    img_array = np.array(img_list)
    return img_array
