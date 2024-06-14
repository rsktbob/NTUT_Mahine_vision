import cv2
import numpy as np
import os
from collections import deque


def transformBinary(img, t):
    # get height, width
    h, w = img.shape[:2]

    # create binary image
    bImg = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            c = np.mean(img[i][j])
            if c <= t:
                bImg[i][j] = 1
            else:
                bImg[i][j] = 0
    return bImg


def transformDistance(img):
    h, w = img.shape
    tImg = np.copy(img)

    def minNeightborDistabce(i, j):
        a = float('inf') if j - 1 < 0 else tImg[i][j-1]
        b = float('inf') if i + 1 >= h else tImg[i+1][j]
        c = float('inf') if j + 1 >= w else tImg[i][j+1]
        d = float('inf') if i - 1 < 0 else tImg[i-1][j]
        return min(a, b, c, d)

    end = False
    while (not end):
        end = True
        for i in range(h):
            for j in range(w):
                if tImg[i][j] > 0:
                    last = tImg[i][j]
                    tImg[i][j] = minNeightborDistabce(i, j) + 1
                    if last != tImg[i][j]:
                        end = False

    return tImg


def transformMedialAxis(img):
    h, w = img.shape
    mImg = np.copy(img)
    mImg = mImg.astype(np.uint8)

    def maxNeightborDistabce(i, j):
        a = 0 if j - 1 < 0 else mImg[i][j-1]
        b = 0 if i + 1 >= h else mImg[i+1][j]
        c = 0 if j + 1 >= w else mImg[i][j+1]
        d = 0 if i - 1 < 0 else mImg[i-1][j]
        return max(a, b, c, d)

    def removeIntensity(i, j):
        def boundCheck(y, x):
            return (y < 0 or y >= h or x < 0 or x >= w) == False

        def neighorBoundCheck(y, x):
            return (y < i - 1 or y > i + 1 or x < j - 1 or x > j + 1) == False

        cur = 0
        x = -1
        y = -1
        p = mImg[i][j]
        record = np.zeros_like(mImg)
        record[i][j] = 1
        for k in range(i-1, i+2):
            for t in range(j-1, j+2):
                if boundCheck(k, t) and mImg[k][t] > 0 and not (k == i and t == j):
                    cur += 1
                    y = k
                    x = t
        mImg[i][j] = 0
        next = 0

        queue = deque([(y, x)])
        while (len(queue) > 0):
            point = queue.popleft()
            k, t = point

            if boundCheck(k, t) and neighorBoundCheck(k, t) and record[k][t] == 0 and mImg[k][t] > 0:
                next += 1
                if next >= cur:
                    break
                record[k][t] = 1
                queue.append((k - 1, t))
                queue.append((k + 1, t))
                queue.append((k, t - 1))
                queue.append((k, t + 1))

        if cur > next:
            mImg[i][j] = p
    maxItensity = np.max(mImg)

    for i in range(1, maxItensity + 1):
        for j in range(h):
            for k in range(w):
                if mImg[j][k] == i and mImg[j][k] < maxNeightborDistabce(j, k):
                    removeIntensity(j, k)

    for i in range(h):
        for j in range(w):
            mImg[i][j] = 255 if mImg[i][j] > 0 else mImg[i][j]

    return mImg


source = "images"
target = "results"

# img1

img1 = cv2.imread(os.path.join(source, "img1.jpg"))
bImg1 = transformBinary(img1, 240)
tImg1 = transformDistance(bImg1)
mImg1 = transformMedialAxis(tImg1)
cv2.imwrite(os.path.join(target, "img1_q1-1.jpg"), tImg1 * 5)
cv2.imwrite(os.path.join(target, "img1_q1-2.jpg"), mImg1)

# img2

img2 = cv2.imread(os.path.join(source, "img2.jpg"))
bImg2 = transformBinary(img2, 240)
tImg2 = transformDistance(bImg2)
mImg2 = transformMedialAxis(tImg2)
cv2.imwrite(os.path.join(target, "img2_q1-1.jpg"), tImg2 * 5)
cv2.imwrite(os.path.join(target, "img2_q1-2.jpg"), mImg2)

# img3

img3 = cv2.imread(os.path.join(source, "img3.jpg"))
bImg3 = transformBinary(img3, 240)
tImg3 = transformDistance(bImg3)
mImg3 = transformMedialAxis(tImg3)
cv2.imwrite(os.path.join(target, "img3_q1-1.jpg"), tImg3 * 5)
cv2.imwrite(os.path.join(target, "img3_q1-2.jpg"), mImg3)

# img4

img4 = cv2.imread(os.path.join(source, "img4.jpg"))
bImg4 = transformBinary(img4, 240)
tImg4 = transformDistance(bImg4)
mImg4 = transformMedialAxis(tImg4)
cv2.imwrite(os.path.join(target, "img4_q1-1.jpg"), tImg4 * 5)
cv2.imwrite(os.path.join(target, "img4_q1-2.jpg"), mImg4)
