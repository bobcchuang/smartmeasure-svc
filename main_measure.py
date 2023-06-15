# -*- coding: utf-8 -*-
import cv2
import time
import os
import numpy as np
from PIL import Image
from utils.cim_measure import *
import argparse


def main(args):
    if not os.path.isdir(args.output_path):
        os.mkdir(args.output_path)

    # read img
    img = np.array(Image.open(args.img_path))
    temp1 = np.array(Image.open(args.temp1_path))
    temp2 = np.array(Image.open(args.temp2_path))
    temp3 = np.array(Image.open(args.temp3_path))
    temp4 = np.array(Image.open(args.temp4_path))

    # get_coordinate
    result = cim_measure(img, temp1, temp2, temp3, temp4, args.match_threshold4, args.match_threshold3,
                         args.clear_threshold, args.ssim_threshold)
    cv2.line(img, (int(result[4]), int(result[5])), (int(result[6]), int(result[7])), (0, 0, 200),
             3)  # (img, start, end, color, width)
    cv2.imwrite(args.output_path + 'result.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cim_measure')
    parser.add_argument('--img_path', type=str, default='./dataset/ABA0D17U1T_5_0001_1013448_0464771.jpg')
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--temp1_path', type=str, default='./output/template_1.jpg')
    parser.add_argument('--temp2_path', type=str, default='./output/template_2.jpg')
    parser.add_argument('--temp3_path', type=str, default='./output/template_3.jpg')
    parser.add_argument('--temp4_path', type=str, default='./output/template_3.jpg')
    parser.add_argument('--match_threshold4', type=float, default=0.1)
    parser.add_argument('--match_threshold3', type=float, default=0.1)
    parser.add_argument('--clear_threshold', type=float, default=5.0)
    parser.add_argument('--ssim_threshold', type=float, default=0.98)
    args, _ = parser.parse_known_args()

    seconds = time.time()
    main(args)
    print(time.time() - seconds)
