from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pandas.core import frame
import scipy.ndimage
import imutils
import argparse
from itertools import islice
import cv2


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

    list_of_img_list = []
    for i in range(0, len(img_list), 300):
        list_of_img_list.append(img_list[i : i + 300])

    frame_delta_list = []
    segment_thresh_lists = []
    segment_frame_list = []
    for listy in list_of_img_list:
        frame_list = []
        thresh_list = []
        for frame in range(0, len(listy), 20):
            try:
                frame_list.append(img_list[frame])
                frame_list.append(img_list[frame+1])
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
        segment_frame_list.append(frame_list)
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
    return img_array, frame_delta_array, segment_thresh_array, len(list_of_img_list), segment_frame_list, np.shape(img_list[0])

frameDeltaGivenPureBackground('videos/01.avi', False)