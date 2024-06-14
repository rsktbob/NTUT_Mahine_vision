import cv2
import numpy as np
import os

# transform img to grayscale img
def transform_grayscale(img):
    img_height, img_width = img.shape[0], img.shape[1]

    # create grayscale img
    gImg = np.zeros((img_height, img_width, 1))

    for i in range(img_height):
        for j in range(img_width):
            gImg[i][j][0] = 0.3 * img[i][j][2] + 0.59 * img[i][j][1] + 0.11 * img[i][j][0]

    return gImg


# transform grayscale img to binary img
def transform_binary(gImg, threshold):
    gImg_height, gImg_width = gImg.shape[0], gImg.shape[1]

    # create binary img
    bImg = np.zeros((gImg_height, gImg_width, 1))

    for i in range(gImg_height):
        for j in range(gImg_width):
            bImg[i][j][0] = 255 if gImg[i][j][0] > threshold else 0

    return bImg


# transform color img to indexed color img
def transform_indexed_color(img, color_table):
    img_height, img_width = img.shape[0], img.shape[1]

    # create indexed color img
    iImg = np.zeros((img_height, img_width, 3))

    getColorDistance = lambda color1, color2: abs(color1[0] - color2[0]) + abs(color1[1] - color2[1]) + abs(color1[2] - color2[2])

    for i in range(img_height):
        for j in range(img_width):
            color = img[i][j][::-1]
            iColor = color_table[0]
            current_distance = getColorDistance(color, iColor)
            for k in range(1, len(color_table)):
                next_distance = getColorDistance(color, color_table[k])
                if current_distance > next_distance:
                    iColor = color_table[k]
                    current_distance = next_distance
            iImg[i][j]= iColor[::-1]

    return iImg


def resize(img, factor):
    img_height, img_width = img.shape[0], img.shape[1]
    rImg_height, rImg_width = int(
        np.round(img_height * factor)), int(np.round(img_width * factor))
    
    # create new size image
    rImg = np.zeros((rImg_height, rImg_width, 3), dtype=np.uint8)

    def getCorrespondPixel(i, j):
        y = int(np.round(-0.5 + (i + 0.5) / factor))
        x = int(np.round(-0.5 + (j + 0.5) / factor))
        return img[y][x]

    for i in range(rImg_height):
        for j in range(rImg_width):
            rImg[i][j] = getCorrespondPixel(i, j)

    return rImg


def resize_bilinear(img, factor):
    img_height, img_width = img.shape[0], img.shape[1]
    rImg_height, rImg_width = int(np.round(img_height * factor)), int(np.round(img_width * factor))

    # create new size image
    rImg = np.zeros((rImg_height, rImg_width, 3), dtype=np.uint8)

    def getCorrespondPixel(i, j):
        y = -0.5 + (i + 0.5) / factor
        x = -0.5 + (j + 0.5) / factor
        y1 = int(y)
        x1 = int(x)
        y2 = min(int(np.round(y + 0.5)), img_height - 1)
        x2 = min(int(np.round(x + 0.5)), img_width - 1)
        q1 = (x2 - x) * img[y1][x1] + (x - x1) * img[y1][x2]
        q1 = img[y1][x1] if x1 == x2 else q1
        q2 = (x2 - x) * img[y2][x1] + (x - x1) * img[y2][x2]
        q2 = img[y2][x1] if x1 == x2 else q2
        return np.round(q1 if y1 == y2 else (y2 - y) * q1 + (y - y1) * q2).astype(np.uint8)

    for i in range(rImg_height):
        for j in range(rImg_width):
            rImg[i][j] = getCorrespondPixel(i, j)

    return rImg

def checkImg(img1, img2):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (img1[i][j][0] != img2[i][j][0]) or (img1[i][j][1] != img2[i][j][1]) or (img1[i][j][2] != img2[i][j][2]):
                print("false img")
                return
    print("true img")


# -----------------------------------------------------------

source = "images"
target = "results"

# img1

img1 = cv2.imread(os.path.join(source, "img1.png"))

# img1_q1-1
gImg1 = transform_grayscale(img1)
cv2.imwrite(os.path.join(target, "img1_q1-1.jpg"), gImg1)
# img1_q1-2
bImg1 = transform_binary(gImg1, 128)
cv2.imwrite(os.path.join(target, "img1_q1-2.jpg"), bImg1)
# img1_q1-3
img1_color_table = [
    (131, 4, 1),
    (6, 130, 2),
    (230, 194, 1),
    (248, 100, 1),
    (7, 122, 115),
    (221, 177, 88),
    (238, 206, 123),
    (229, 194, 105),
    (51, 18, 1),
    (184, 165, 134),
    (216, 109, 85),
    (145, 177, 186),
    (205, 154, 62),
    (252, 243, 50),
    (90, 66, 42),
    (127, 220, 116)
]
iImg1 = transform_indexed_color(img1, img1_color_table)
cv2.imwrite(os.path.join(target, "img1_q1-3.jpg"), iImg1)

# img1_q2-1
hImg1_1 = resize(img1, 0.5)
cv2.imwrite(os.path.join(target, "img1_q2-1-half.jpg"), hImg1_1)
dImg1_1 = resize(img1, 2)
cv2.imwrite(os.path.join(target, "img1_q2-1-double.jpg"), dImg1_1)
# img1_q2-2
hImg1_2 = resize_bilinear(img1, 0.5)
cv2.imwrite(os.path.join(target, "img1_q2-2-half.jpg"), hImg1_2)
dImg1_2 = resize_bilinear(img1, 2)
cv2.imwrite(os.path.join(target, "img1_q2-2-double.jpg"), dImg1_2)

# -----------------------------------------------------------

# img2

img2 = cv2.imread(os.path.join(source, "img2.png"))

# img2_q1-1
gImg2 = transform_grayscale(img2)
cv2.imwrite(os.path.join(target, "img2_q1-1.jpg"), gImg2)
# img2_q1-2
bImg2 = transform_binary(gImg2, 128)
cv2.imwrite(os.path.join(target, "img2_q1-2.jpg"), bImg2)
# img2_q1-3
img2_color_table = [
    (240, 222, 208),
    (216, 197, 172),
    (226, 210, 195),
    (73, 53, 27),
    (89, 64, 37),
    (28, 10, 6),
    (195, 175, 149),
    (247, 200, 182),
    (197, 121, 108),
    (98, 101, 75),
    (126, 107, 93),
    (100, 83, 58),
    (59, 43, 28),
    (159, 133, 106),
    (85, 61, 47),
    (187, 170, 156),
]
iImg2 = transform_indexed_color(img2, img2_color_table)
cv2.imwrite(os.path.join(target, "img2_q1-3.jpg"), iImg2)

# img2_q2-1
hImg2_1 = resize(img2, 0.5)
cv2.imwrite(os.path.join(target, "img2_q2-1-half.jpg"), hImg2_1)
dImg2_1 = resize(img2, 2)
cv2.imwrite(os.path.join(target, "img2_q2-1-double.jpg"), dImg2_1)
# img2_q2-2
hImg2_2 = resize_bilinear(img2, 0.5)
cv2.imwrite(os.path.join(target, "img2_q2-2-half.jpg"), hImg2_2)
dImg2_2 = resize_bilinear(img2, 2)
cv2.imwrite(os.path.join(target, "img2_q2-2-double.jpg"), dImg2_2)

# -----------------------------------------------------------

# img3

img3 = cv2.imread(os.path.join(source, "img3.png"))

# img3_q1-1
gImg3 = transform_grayscale(img3)
cv2.imwrite(os.path.join(target, "img3_q1-1.jpg"), gImg3)
# img3_q1-2
bImg3 = transform_binary(gImg3, 128)
cv2.imwrite(os.path.join(target, "img3_q1-2.jpg"), bImg3)
# img3_q1-3
img3_color_table = [
    (115, 159, 60),
    (162, 201, 113),
    (164, 209, 239),
    (245, 248, 247),
    (28, 29, 69),
    (232, 176, 152),
    (101, 195, 202),
    (225, 118, 70),
    (7, 10, 11),
    (210, 48, 73),
    (239, 249, 125),
    (146, 201, 189),
    (63, 173, 111),
    (78, 114, 182),
    (242, 198, 106),
    (45, 140, 115)
]
iImg3 = transform_indexed_color(img3, img3_color_table)
cv2.imwrite(os.path.join(target, "img3_q1-3.jpg"), iImg3)

# img3_q2-1
hImg3_1 = resize(img3, 0.5)
cv2.imwrite(os.path.join(target, "img3_q2-1-half.jpg"), hImg3_1)
dImg3_1 = resize(img3, 2)
cv2.imwrite(os.path.join(target, "img3_q2-1-double.jpg"), dImg3_1)
# img3_q2-2
hImg3_2 = resize_bilinear(img3, 0.5)
cv2.imwrite(os.path.join(target, "img3_q2-2-half.jpg"), hImg3_2)
dImg3_2 = resize_bilinear(img3, 2)
cv2.imwrite(os.path.join(target, "img3_q2-2-double.jpg"), dImg3_2)