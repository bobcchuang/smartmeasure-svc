# -*- coding: utf-8 -*-
import cv2
import time
import os
import numpy as np
from PIL import Image
from utils.cut_img import *


def main():
    img_path = './dataset/ABA0D17U1T_5_0001_1013448_0464771.jpg'
    output_path = './output/'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # read img
    img = np.array(Image.open(img_path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get_template
    # test1
    # center_1, center_2 = [1388, 558], [1101, 1031]
    # x_radius_1, y_radius_1 = 180, 180
    # x_radius_2, y_radius_2 = 150, 150

    # test2
    center_1, center_2 = [1034, 1031], [1168, 1031]
    x_radius_1, y_radius_1 = 40, 100
    x_radius_2, y_radius_2 = 40, 100

    temp0, temp1, temp2 = get_template(img, center_1, x_radius_1, y_radius_1, center_2, x_radius_2, y_radius_2)
    cv2.imwrite(output_path + 'template_3.jpg', cv2.cvtColor(temp0, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path + 'template_1.jpg', cv2.cvtColor(temp1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path + 'template_2.jpg', cv2.cvtColor(temp2, cv2.COLOR_RGB2BGR))

    # get_one_template
    # center_1 = [1388, 558]
    # x_radius_1, y_radius_1 = 180, 180
    # temp = get_one_template(img, center_1, x_radius_1, y_radius_1)
    # cv2.imwrite(output_path + 'template.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    seconds = time.time()
    main()
    print(time.time() - seconds)
