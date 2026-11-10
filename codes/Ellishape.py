import cv2
import numpy as np
import os
from pyefd import elliptic_fourier_descriptors
import pandas as pd

COLORS = {
        'head': (0, 0, 255),
        'torso': (0, 255, 0),
        'left_arm': (255, 0, 0),
        'right_arm': (255, 255, 0),
        'left_leg': (0, 255, 255),
        'right_leg': (255, 0, 255),
        'dress': (128, 128, 128),
        'background': (255, 255, 255)}

N = 30

def extract_part_contours(parsing_result, runner):
    contours_dict = {}
    for class_name, class_id in runner.PARTS.items():
        if class_name == 'background':
            mask = np.uint8(parsing_result != class_id) * 255
        else:
            mask = np.uint8(np.isin(parsing_result, class_id)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(mask,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)
        else: contour = None
        contours_dict[class_name] = contour

    return contours_dict

def get_fourier(contour, runner):
    n = runner.HARMONICS
    contour = contour.reshape(-1, 2)
    if len(contour) < 3:
        return None
    efd = elliptic_fourier_descriptors(contour, order=n, normalize=True)
    return efd

def save_contour_image(img, contour_dict, save_path):
    result = img.copy()

    overlay = result.copy()

    for class_name, contour in contour_dict.items():
        if contour is None:
            continue

        color = COLORS.get(class_name, (255, 255, 255))
        cv2.drawContours(overlay, [contour], -1, color, thickness=-1)

    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, result, 1-alpha, 0)

    color = COLORS['background']
    cv2.drawContours(result, [contour_dict['background']], -1, color, thickness=1)

    cv2.imwrite(save_path, result)
    return True