# version:0.0.3
# 0.0.3 --> add match_dist2 in cim_measure
# 0.0.4 --> add def clear_check
# 0.0.5 --> add def cal_ssim . def multi_cal_ssim / modify def cim_measure

import cv2
import numpy as np


# import pandas as pd
# from matplotlib import pyplot as plt
# import os
# import time
# from math import sqrt


def match_template(img, template):
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    w = template.shape[1]
    h = template.shape[0]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    center = (int(top_left[0] + w / 2), int(top_left[1] + h / 2))
    return center, top_left, bottom_right, min_val


def get_clear_score(img):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.Laplacian(img2gray, cv2.CV_64F)
    clear_score = res.var()
    return clear_score


def cal_ssim(im1, im2):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    #     assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    #     s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12
    return ssim


def multi_cal_ssim(im1, im2):
    assert len(im1.shape) == 3 and len(im2.shape) == 3
    assert im1.shape == im2.shape
    temp_sum = 0
    for i in range(im1.shape[2]):
        temp_sum += cal_ssim(im1[:, :, i], im2[:, :, i])
    return temp_sum / im1.shape[2]


def cim_measure(img, template_1, template_2, template_3, template_4='',
                match_threshold4=0.1, match_threshold3=0.1,
                clear_threshold=8, ssim_threshold=0.98):
    result = np.full(8, -1.0)
    if len(template_4) <= 0:
        # print('----no template_4----')
        template_4 = template_3.copy()

    # check template_4 
    center_4, top_left_4, bottom_right_4, min_val_4 = match_template(img, template_4)
    result[0] = min_val_4

    if min_val_4 >= match_threshold4:
        return result

    # check clearness of area_4 in img
    temp_img = img[top_left_4[1]:bottom_right_4[1], top_left_4[0]:bottom_right_4[0]]
    #     final_clear_threshold = get_clear_score(template_4) + clear_threshold
    result[1] = get_clear_score(temp_img)
    if result[1] < clear_threshold:
        return result

    # check color_dist between template4 and temp_img
    result[2] = multi_cal_ssim(temp_img, template_4)
    if result[2] < ssim_threshold:
        return result

    # match template_3
    center_3, top_left_3, bottom_right_3, min_val_3 = match_template(temp_img, template_3)
    result[3] = min_val_3
    if min_val_3 >= match_threshold3:
        return result
    temp_img2 = temp_img[top_left_3[1]:bottom_right_3[1], top_left_3[0]:bottom_right_3[0]]

    # match template_1 .template_2 
    center_1, top_left_1, bottom_right_1, min_val_1 = match_template(temp_img2, template_1)
    center_2, top_left_2, bottom_right_2, min_val_2 = match_template(temp_img2, template_2)

    real_center_1 = np.array(top_left_4) + np.array(top_left_3) + np.array(center_1)
    real_center_2 = np.array(top_left_4) + np.array(top_left_3) + np.array(center_2)
    result[4] = real_center_1[0]
    result[5] = real_center_1[1]
    result[6] = real_center_2[0]
    result[7] = real_center_2[1]
    return result


def auoDrawBbox(image_bgr, bbox_min, bbox_max, line_color, line_width=2):
    cv2.rectangle(image_bgr, bbox_min, bbox_max, line_color, line_width)
    return image_bgr


def text2image(image, xy, label, font_scale=0.5, thickness=1, font_color=(0, 0, 0),
               font_face=cv2.FONT_HERSHEY_COMPLEX, background_color=(0, 255, 0)):
    label_size = cv2.getTextSize(label, font_face, font_scale, thickness)
    _x1 = xy[0]  # bottomleft x of text
    _y1 = xy[1]  # bottomleft y of text
    _x2 = xy[0] + label_size[0][0]  # topright x of text
    _y2 = xy[1] - label_size[0][1]  # topright y of text
    cv2.rectangle(image, (_x1, _y1), (_x2, _y2), background_color, cv2.FILLED)  # text background
    cv2.putText(image, label, (_x1, _y1), font_face, font_scale, font_color,
                thickness, cv2.LINE_AA)
    return image
