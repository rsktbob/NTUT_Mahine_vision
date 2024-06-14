import cv2
import numpy as np
import os
import math
from collections import deque
import random


def filter(img, filter_size, filter_strategy):
    h, w = img.shape[:2]
    r = filter_size // 2
    nImg = np.zeros((h, w))

    def get_intensity(y, x):
        return img[y][x] if y >= 0 and y < h and x >= 0 and x < w else 0

    record = deque()
    k = 0
    for i in range(-r, r+1):
        record.append(deque())
        for j in range(-r-1, r):
            record[k].append(get_intensity(i, j))
        k += 1
    right = 1
    i = 0
    j = 0

    while True:
        for k in range(len(record)):
            del record[k][0 if right == 1 else len(record[k]) - 1]
        t = 0
        for k in range(i-r, i+r+1):
            if right == 1:
                record[t].append(get_intensity(k, j+right*r))
            else:
                record[t].appendleft(get_intensity(k, j+right*r))
            t += 1
        nImg[i][j] = filter_strategy(record)
        j += right
        if (j > w-1 and right == 1) or (j < 0 and right == -1):
            j -= right
            i += 1
            if i >= h:
                break
            del record[0]
            record.append(deque())
            for k in range(j-r, j+r+1):
                record[len(record)-1].append(get_intensity(i+1, k))
            nImg[i][j] = filter_strategy(record)
            right = -right
            j += right
    return nImg


def mean(matrix):
    h, w = len(matrix), len(matrix[0])
    sum = 0
    for i in range(h):
        for j in range(w):
            sum += matrix[i][j]
    sum = np.round(sum / (h * w))
    return sum


def median(matrix):
    flattene_matrix = [i for row in matrix for i in row]
    flattene_matrix = sorted(flattene_matrix)
    return flattene_matrix[len(flattene_matrix) // 2]


def gaussian(matrix):
    h, w = len(matrix), len(matrix[0])
    r = h // 2
    sigma = 1
    weight_sum = 0
    sum = 0

    for i in range(h):
        for j in range(w):
            weight = (1 / (2 * math.pi * (sigma ** 2))) * (math.e **
                                                           (-((i-r) ** 2 + (j-r) ** 2) / (2 * (sigma ** 2))))
            sum += weight * matrix[i][j]
            weight_sum += weight
    sum = np.round(sum / weight_sum)
    return sum

source = "images"
target = "results"

# img1

img1 = cv2.imread(os.path.join(source, "img1.jpg"), cv2.IMREAD_GRAYSCALE)

img1_mean_3 = filter(img1, 3, mean)
img1_mean_7 = filter(img1, 7, mean)

cv2.imwrite(os.path.join(target, "img1_q1_3.jpg"), img1_mean_3)
cv2.imwrite(os.path.join(target, "img1_q1_7.jpg"), img1_mean_7)

img1_median_3 = filter(img1, 3, median)
img1_median_7 = filter(img1, 7, median)

cv2.imwrite(os.path.join(target, "img1_q2_3.jpg"), img1_median_3)
cv2.imwrite(os.path.join(target, "img1_q2_7.jpg"), img1_median_7)

img1_gaussian = filter(img1, 5, gaussian)

cv2.imwrite(os.path.join(target, "img1_q3.jpg"), img1_gaussian)

# img2

img2 = cv2.imread(os.path.join(source, "img2.jpg"), cv2.IMREAD_GRAYSCALE)

img2_mean_3 = filter(img2, 3, mean)
img2_mean_7 = filter(img2, 7, mean)

cv2.imwrite(os.path.join(target, "img2_q1_3.jpg"), img2_mean_3)
cv2.imwrite(os.path.join(target, "img2_q1_7.jpg"), img2_mean_7)

img2_median_3 = filter(img2, 3, median)
img2_median_7 = filter(img2, 7, median)

cv2.imwrite(os.path.join(target, "img2_q2_3.jpg"), img2_median_3)
cv2.imwrite(os.path.join(target, "img2_q2_7.jpg"), img2_median_7)

img2_gaussian = filter(img2, 5, gaussian)

cv2.imwrite(os.path.join(target, "img2_q3.jpg"), img2_gaussian)


# img3

img3 = cv2.imread(os.path.join(source, "img3.jpg"), cv2.IMREAD_GRAYSCALE)

img3_mean_3 = filter(img3, 3, mean)
img3_mean_7 = filter(img3, 7, mean)

cv2.imwrite(os.path.join(target, "img3_q1_3.jpg"), img3_mean_3)
cv2.imwrite(os.path.join(target, "img3_q1_7.jpg"), img3_mean_7)

img3_median_3 = filter(img3, 3, median)
img3_median_7 = filter(img3, 7, median)

cv2.imwrite(os.path.join(target, "img3_q2_3.jpg"), img3_median_3)
cv2.imwrite(os.path.join(target, "img3_q2_7.jpg"), img3_median_7)

img3_gaussian = filter(img3, 5, gaussian)

cv2.imwrite(os.path.join(target, "img3_q3.jpg"), img3_gaussian)