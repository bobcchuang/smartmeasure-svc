import time
import cv2
import numpy as np


def get_template(img, center_1, x_radius_1, y_radius_1, center_2, x_radius_2, y_radius_2):
    temp1 = img[(center_1[1] - y_radius_1):(center_1[1] + y_radius_1 + 1),
            (center_1[0] - x_radius_1):(center_1[0] + x_radius_1 + 1), :]
    temp2 = img[(center_2[1] - y_radius_2):(center_2[1] + y_radius_2 + 1),
            (center_2[0] - x_radius_2):(center_2[0] + x_radius_2 + 1), :]

    x_min, x_max = min(center_1[0] - x_radius_1, center_2[0] - x_radius_2) - 10, max(center_1[0] + x_radius_1,
                                                                                     center_2[0] + x_radius_2) + 10
    y_min, y_max = min(center_1[1] - y_radius_1, center_2[1] - y_radius_2) - 10, max(center_1[1] + y_radius_1,
                                                                                     center_2[1] + y_radius_2) + 10
    temp0 = img[y_min:y_max, x_min:x_max, :]

    return temp0, temp1, temp2


def get_one_template(img, center_1, x_radius_1, y_radius_1):
    cut_img = img[(center_1[1] - y_radius_1):(center_1[1] + y_radius_1 + 1),
              (center_1[0] - x_radius_1):(center_1[0] + x_radius_1 + 1), :]

    return cut_img
