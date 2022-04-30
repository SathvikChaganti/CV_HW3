# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os


def template_matching_ssd(src, temp):
    h, w = src.shape
    ht, wt = temp.shape

    score = np.empty((h - ht, w - wt))

    for dy in range(0, h - ht):
        for dx in range(0, w - wt):
            diff = (src[dy:dy + ht, dx:dx + wt] - temp) ** 2
            score[dy, dx] = diff.sum()

    pt = np.unravel_index(score.argmin(), score.shape)

    return pt[1], pt[0]


def main():
    folder_dir = 'images'
    for image in os.listdir(folder_dir):
        img = cv2.imread("images/"+image)
        temp = cv2.imread("template.png")
        img2 = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        temp = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

        h, w = temp.shape

        pt = template_matching_ssd(gray, temp)

        i = cv2.rectangle(img, (pt[0], pt[1]), (pt[0] + w, pt[1] + h), (0, 0, 200), 3)
        cv2.imwrite("output/"+image, img)


if __name__ == "__main__":
    main()
