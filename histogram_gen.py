from typing import List
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import imutils
import argparse
from itertools import islice
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


def frameDeltaGivenPureBackground(video, show_imgs):
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

    print("number of frames", len(img_list))

    list_of_img_list = []
    for i in range(0, len(img_list), 600):
        list_of_img_list.append(img_list[i : i + 600])

    print("number of segments", len(list_of_img_list))

    frame_delta_list = []
    segment_thresh_lists = []
    for list in list_of_img_list:
        thresh_list = []
        for frame in range(len(list)):
            try:
                frameDelta = cv2.absdiff(img_list[frame], img_list[frame + 1])
                frame_delta_list.append(frameDelta)

                thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
                # dilate the thresholded image to fill in holes
                thresh = cv2.dilate(thresh, None, iterations=2)
                thresh = np.array(thresh).ravel()
                # print(thresh.shape)
                thresh_list.append(thresh)
            except:
                pass
        segment_thresh_lists.append(thresh_list)

    if show_imgs:
        cv2.imshow("im1", img_list[0])
        cv2.imshow("im2", img_list[60])
        cv2.imshow("Frame Delta", frameDelta)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey()

    img_array = np.array(img_list)
    frame_delta_array = np.array(frame_delta_list)
    thresh_array = np.array(thresh_list)
    segment_thresh_array = np.array(segment_thresh_lists)
    return img_array, frame_delta_array, segment_thresh_array, len(list_of_img_list)
