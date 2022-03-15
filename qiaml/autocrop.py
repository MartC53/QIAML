# -*- coding: utf-8 -*-
"""
Contains the several useful image analysis functions, specifically crop and
apply_contrast.

Authors: Coleman Martin and Zoe Kaputa

Modified on Sun Mar 13,
assertions wrote for tests
Xuetao.
"""
import cv2 as cv2
import numpy as np


def crop(path=None):
    """
    Returns a cropped version of the image with the path path.

    Parameters:
    - path: path to the image being cropped

    Returns:
    - cropped: image which is the cropped version of the image with the path
               path
    """
    assert type(path) == str, 'The path should be in string format!'
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    assert np.sum(img) != None, 'the path is not right or there is no such a file. Check path or file name.'
    assert img.shape[0:3] != None and img.ndim == 3, 'The image is not in right format. Image should have three diamensions'
    
    assert 2000> img.shape[0] >200 and 2000> img.shape[1] >200, 'The image has unusual size. Please check the image imported.'

    # leave only green color
    img[:, :, 0] = 0
    img[:, :, 2] = 0

    # convert to gray scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # more contrast between foreground and background
    contrasted_img = apply_contrast(gray_img)

    # erode image
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(contrasted_img, kernel, iterations=15)

    # located contours
    contours = _locate_contours(eroded_img)

    img_contours = np.zeros(img.shape)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    # determine cropped image based on contours
    crop_box = _determine_cropped_image_box(img, contours)

    # crop the original image
    cropped = img[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]

    return cropped


def apply_contrast(img):
    """
    Returns a contrasted version of img.

    Returns:
    - contrasted_img: contrasted version of img
    """

    contrast_threshold = 2
    grid_size = 2
    alpha = 3  # (1.0-3.0)
    beta = 0  # (0-100)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=contrast_threshold,
                            tileGridSize=(grid_size, grid_size))
    clahe_img = clahe.apply(img)

    adjusted = cv2.convertScaleAbs(clahe_img, alpha=alpha, beta=beta)

    return adjusted


def _locate_contours(img):
    """
    Returns the substantial contours in img.

    Parameters:
    - img: the image being analyzed

    Returns:
    - substantial_contours: the substantial contours in img
    """

    min_threshold = 75
    threshold_output = 255
    min_countour_area = 15000

    _, threshold = cv2.threshold(img, min_threshold,
                                 threshold_output,
                                 cv2.THRESH_BINARY)

    # dilated = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,
    #                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
    #                            (10, 10)))

    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    substantial_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_countour_area:
            substantial_contours.append(contour)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    return substantial_contours


def _determine_cropped_image_box(img, contours):
    """
    Returns the pixel box including all contours in contours.

    Parameters:
    - img: the image being analyzed
    - contours: a list of countours thats locations should be included in the
                outputted box size

    Returns:
    - crop_box: the pixel box including all contours in contours in form [left,
                top, right, bottom]
    """
    # https://stackoverflow.com/questions/37803903/opencv-and-python-for-auto-cropping
    crop_box = [-1, -1, -1, -1]
    for contour in contours:
        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
        if crop_box[0] < 0:
            crop_box = [contour_x, contour_y, contour_x + contour_w,
                        contour_y + contour_h]
        elif contour_x > np.shape(img)[0] / 2:
            crop_box[0] = min(contour_x, crop_box[0])
            crop_box[1] = min(contour_y, crop_box[1])
            crop_box[2] = max(contour_x + contour_w, crop_box[2])
            crop_box[3] = max(contour_y + contour_h, crop_box[3])
    
    # add bounding space
    crop_box[0] = max(0, crop_box[0] - 50)
    crop_box[1] = max(0, crop_box[1] - 50)
    crop_box[2] = min(np.shape(img)[0], crop_box[2] + 100)
    crop_box[3] = min(np.shape(img)[1], crop_box[3] + 100)

    return crop_box
