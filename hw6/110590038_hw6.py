import cv2
import numpy as np
import math
import os


def get_intensity(i, j, img):
    h, w = img.shape[:2]
    return 0 if i < 0 or i >= h or j < 0 or j >= w else img[i][j]

def filter(img, kernel):
    h, w = img.shape[:2]
    size = int(kernel.shape[0] / 2)
    nImg = np.zeros_like(img)
    
    def get_sum_intensity(i, j):
        sum = 0
        weight_sum = 0
        for k in range(i-size, i+size+1):
            for t in range(j-size, j+size):
                sum += get_intensity(k, t, img) * kernel[k-i+1][t-j+1]
                weight_sum += kernel[k-i+1][t-j+1]

        sum /= 1 if weight_sum == 0 else weight_sum
        return sum
    
    for i in range(h):
        for j in range(w):
            nImg[i][j] = get_sum_intensity(i, j)
    
    return nImg


def integrate_sobel(sobelX_img, sobelY_img):
    h, w = sobelX_img.shape[:2]
    nImg = np.zeros_like(sobelX_img)
    angle = np.zeros_like(sobelX_img)

    # integrate
    for i in range(h):
        for j in range(w):
            y = sobelY_img[i][j]
            x = sobelX_img[i][j]
            sum = math.sqrt(x ** 2 + y ** 2)
            nImg[i][j] = sum
            angle[i][j] = (math.degrees(math.atan2(y, x)) + 180) % 180

    return nImg, angle


def non_maximum_suppression(img, angle):
    h, w = img.shape[:2]
    nImg = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                q = get_intensity(i, j+1, img)
                r = get_intensity(i, j-1, img)
            elif (22.5 <= angle[i][j] < 67.5):
                q = get_intensity(i+1, j-1, img)
                r = get_intensity(i-1, j+1, img)
            elif (67.5 <= angle[i][j]< 112.5):
                q = get_intensity(i+1, j, img)
                r = get_intensity(i-1, j, img)
            elif (112.5 <= angle[i][j] < 157.5):
                q = get_intensity(i-1, j-1, img)
                r = get_intensity(i+1, j+1, img)

            if img[i][j] >= q and img[i][j] >= r:
                nImg[i][j] = img[i][j]
    
    return nImg
            

def double_threshold(img):
    h, w = img.shape[:2]
    low_threshold = 70
    high_threshold = 140
    edges = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            if img[i][j] > low_threshold and img[i][j] < high_threshold:
                edges[i][j] = 1
            elif img[i][j] >= high_threshold:
                edges[i][j] = 2

    return edges


def hysteresis(angle, edges):
    h, w = edges.shape[:2]
    nImg = np.zeros_like(edges)
    record = np.zeros_like(edges)

    for i in range(h):
        for j in range(w):
            if record[i][j] == 0 and edges[i][j] == 2:
                stack = [(i, j)]
                while len(stack) > 0:
                    y, x = stack.pop()
                    nImg[y][x] = 255
                    record[y][x] = 1
                    neighbor = [0, 0]
                    if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                        neighbor[0] = (i+1, j)
                        neighbor[1] = (i-1, j)
                    elif (22.5 <= angle[i][j] < 67.5):
                        neighbor[0] = (i-1, j-1)
                        neighbor[1] = (i+1, j+1)
                    elif (67.5 <= angle[i][j]< 112.5):
                        neighbor[0] = (i, j+1)
                        neighbor[1] = (i, j-1)
                    elif (112.5 <= angle[i][j] < 157.5):
                        neighbor[0] = (i+1, j-1)
                        neighbor[1] = (i-1, j+1)
                    for k in neighbor:
                        y, x = k
                        if y < 0 or y >= h or x < 0 or x >= w or edges[y][x] != 1 or record[y][x] != 0:
                            continue
                        record[y][x] = 1
                        stack.append(k)
      
    return nImg

def canny(img, gaussion_kernel, sobelX_kernel, sobelY_kernel):
    gaussion_img = filter(img, gaussion_kernel)
    sobelX_img = filter(gaussion_img, sobelX_kernel)
    sobelY_img = filter(gaussion_img, sobelY_kernel)
    sobel_img, angle = integrate_sobel(sobelX_img, sobelY_img)
    sobel_img_supperession = non_maximum_suppression(sobel_img, angle)
    edges = double_threshold(sobel_img_supperession)
    canny_img = hysteresis(angle, edges)
    return canny_img

source = "images"
target = "results"


gaussion_kernel = np.zeros((5, 5), dtype=np.float32)
sigma = 1.5
for i in range(gaussion_kernel.shape[0]):
    for j in range(gaussion_kernel.shape[0]):
        gaussion_kernel[i][j] = 1 / (2 * math.pi * (sigma ** 2)) * \
            (math.e ** (-((i-1)**2 + (j-1)**2) / (2 * (sigma ** 2))))
        
sobelX_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
])

sobelY_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
])

# img1

img1 = cv2.imread(os.path.join(source, "img1.jpg"), cv2.IMREAD_GRAYSCALE)
canny_img1 = canny(img1, gaussion_kernel, sobelX_kernel, sobelY_kernel)
cv2.imwrite(os.path.join(target, "canny_img1.jpg"), canny_img1)

# img2

img2 = cv2.imread(os.path.join(source, "img2.jpg"), cv2.IMREAD_GRAYSCALE)
canny_img2 = canny(img2, gaussion_kernel, sobelX_kernel, sobelY_kernel)
cv2.imwrite(os.path.join(target, "canny_img2.jpg"), canny_img2)

# img3

img3 = cv2.imread(os.path.join(source, "img3.jpg"), cv2.IMREAD_GRAYSCALE)
canny_img3 = canny(img3, gaussion_kernel, sobelX_kernel, sobelY_kernel)
cv2.imwrite(os.path.join(target, "canny_img3.jpg"), canny_img3)