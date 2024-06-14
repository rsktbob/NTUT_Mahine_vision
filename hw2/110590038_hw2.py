import cv2
import numpy as np
import os

def transformBinary(img, t):
    # get height, width
    h, w = img.shape[:2]

    # create binary image
    bImg = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            c = np.mean(img[i][j])
            if c <= t:
                bImg[i][j] = 0
            else:
                bImg[i][j] = 255
    return bImg


def connected_components_4(bImg, colors):
    # get height, width
    h, w = bImg.shape

    # create connect components 4 image
    nImg = np.zeros((h, w, 3), dtype=np.uint8)

    # Record whether it has been visited
    visited = np.zeros((h, w), dtype=np.uint8)

    k = -1

    def visit(y, x, record):
        if y < 0 or x < 0 or y >= h or x >= w or bImg[y][x] == 255 or visited[y][x] == 1:
            return []
        if len(record) < 60:
            record.append((y, x))
        nImg[y][x] = colors[k]
        visited[y][x] = 1
        return [(y, x-1), (y, x+1), (y-1, x), (y+1, x)]

    for i in range(h):
        for j in range(w):
            stack = []
            record = []
            size = 0
            isGraph = False
            if bImg[i][j] == 0 and visited[i][j] == 0:
                k += 1
                isGraph = True
                stack.append((i, j))
            while (len(stack) > 0):
                y, x = stack.pop()
                stack.extend(visit(y, x, record))
            if len(record) < 60 and isGraph:
                k -= 1
                for y, x in record:
                    nImg[y][x] = 0
    
    return nImg


def connected_components_8(bImg, colors):
    # get height, width
    h, w = bImg.shape

    # create connect components 4 image
    nImg = np.zeros((h, w, 3), dtype=np.uint8)

    # Record whether it has been visited
    visited = np.zeros((h, w), dtype=np.uint8)

    k = -1

    def visit(y, x, record):
        if y < 0 or x < 0 or y >= h or x >= w or bImg[y][x] == 255 or visited[y][x] == 1:
            return []
        if len(record) < 60:
            record.append((y, x))
        nImg[y][x] = colors[k]
        visited[y][x] = 1
        return [(y, x-1), (y, x+1), (y-1, x), (y+1, x), (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)]

    for i in range(h):
        for j in range(w):
            stack = []
            record = []
            size = 0
            isGraph = False
            if bImg[i][j] == 0 and visited[i][j] == 0:
                k += 1
                isGraph = True
                stack.append((i, j))
            while (len(stack) > 0):
                y, x = stack.pop()
                stack.extend(visit(y, x, record))
            if len(record) < 60 and isGraph:
                k -= 1
                for y, x in record:
                    nImg[y][x] = 0
    
    return nImg


source = "images"
target = "results"

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (255, 128, 0),
    (0, 255, 128),
    (255, 0, 128),
    (128, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (128, 64, 0),
    (0, 128, 64),
    (128, 0, 64),
    (64, 128, 0),
    (0, 64, 128),
    (64, 0, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (192, 0, 0),
    (0, 192, 0),
    (0, 0, 192),
    (192, 192, 0),
    (0, 192, 192),
    (192, 0, 192),
    (192, 64, 0),
    (0, 192, 64),
    (192, 0, 64),
]

# img1

img1 = cv2.imread(os.path.join(source, "img1.png"))
bImg1 = transformBinary(img1, 235)

img1_4 = connected_components_4(bImg1, colors)
cv2.imwrite(os.path.join(target, "img1_4.jpg"), img1_4)
img1_8 = connected_components_8(bImg1, colors)
cv2.imwrite(os.path.join(target, "img1_8.jpg"), img1_8)

# img2

img2 = cv2.imread(os.path.join(source, "img2.png"))
bImg2 = transformBinary(img2, 190)

img2_4 = connected_components_4(bImg2, colors)
cv2.imwrite(os.path.join(target, "img2_4.jpg"), img2_4)
img2_8 = connected_components_8(bImg2, colors)
cv2.imwrite(os.path.join(target, "img2_8.jpg"), img2_8)

# img3

img3 = cv2.imread(os.path.join(source, "img3.png"))
bImg3 = transformBinary(img3, 245)

img3_4 = connected_components_4(bImg3, colors)
cv2.imwrite(os.path.join(target, "img3_4.jpg"), img3_4)
img3_8 = connected_components_8(bImg3, colors)
cv2.imwrite(os.path.join(target, "img3_8.jpg"), img3_8)

# img4

img4 = cv2.imread(os.path.join(source, "img4.png"))
bImg4 = transformBinary(img4, 229)

img4_4 = connected_components_4(bImg4, colors)
cv2.imwrite(os.path.join(target, "img4_4.jpg"), img4_4)
img4_8 = connected_components_8(bImg4, colors)
cv2.imwrite(os.path.join(target, "img4_8.jpg"), img4_8)
