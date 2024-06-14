import cv2
import numpy as np
import os
from collections import deque

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
]

drawing = False

def paint_image(image):
    h, w = image.shape[:2]
    drawing = False
    color_pos = 0
    record_labels = np.zeros((h, w), dtype=np.int16)
    image_copy = np.copy(image)

    def paint(x, y):
        for i in range(y-5, y+6):
            for j in range(x-5, x+6):
                k = min(max(i, 0), h-1)
                t = min(max(j, 0), w-1)
                record_labels[k][t] = color_pos + 1
                image_copy[k][t] = colors[color_pos]


    def mark_point(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            paint(x, y)
            cv2.imshow("Mark Image", image_copy)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                paint(x, y)
                cv2.imshow("Mark Image", image_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.imshow("Mark Image", image_copy)    
    cv2.setMouseCallback("Mark Image", mark_point)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("1"):
            color_pos = 0
        elif key == ord("2"):
            color_pos = 1
        elif key == ord("3"):
            color_pos = 2
        elif key == ord("4"):
            color_pos = 3
        elif key == ord("5"):
            color_pos = 4 
        elif key == ord("6"):
            color_pos = 5
        elif key == ord("7"):
            color_pos = 6
        elif key == ord("8"):
            color_pos = 7
        elif key == ord("9"):
            color_pos = 8    
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return image_copy, record_labels


def watershed_segmentation(image, image_copy, record_labels):
    h, w = record_labels.shape[:2]
    label_neighbors = {}

    def get_neighbor_points(y, x):
        return (y, x-1), (y-1, x), (y, x+1), [y+1, x]
    
    def boundary_check(y, x):
        return x >= 0 and x < w and y >= 0 and y < h


    # label find
    for i in range(h):
        for j in range(w):
            if record_labels[i][j] == 0:
                neighbors = get_neighbor_points(i, j)
                for neighbor in neighbors:
                    y = neighbor[0]
                    x = neighbor[1]
                    if boundary_check(y, x) and record_labels[y][x] > 0:
                        record_labels[i][j] = -2
                        label = record_labels[y][x]
                        if label not in label_neighbors:
                            label_neighbors[label] = deque()
                        label_neighbors[label].append((i, j))
                        break

    # segementation
    for label in label_neighbors:
        # flood
        while len(label_neighbors[label]) != 0:
            y, x = label_neighbors[label].popleft()
            neighbors = get_neighbor_points(y, x)
            isEdge = False
            count = 0
            for neighbor in neighbors:
                k = neighbor[0]
                t = neighbor[1]
                if boundary_check(k, t):
                    if record_labels[k][t] > 0 and record_labels[k][t] != label:
                        isEdge = True
                        record_labels[y][x] = -1
                        for i in range(count):
                            label_neighbors[label].pop()
                        break
                    elif record_labels[k][t] == 0:
                        record_labels[k][t] = -2
                        label_neighbors[label].append(neighbor)
                        count += 1

            if isEdge == False:
                record_labels[y][x] = label
                image_copy[y][x] = colors[int(label) - 1]

source = "images"
target = "results"

# Img1

img1 = cv2.imread(os.path.join(source, "img1.jpg"))

img1_q1, img1_labels = paint_image(img1)

watershed_segmentation(img1, img1_q1, img1_labels)
cv2.imwrite(os.path.join(target, "img1_q1.jpg"), img1_q1)


# Img2

img2 = cv2.imread(os.path.join(source, "img2.jpg"))

img2_q1, img2_labels = paint_image(img2)

watershed_segmentation(img2, img2_q1, img2_labels)
cv2.imwrite(os.path.join(target, "img2_q1.jpg"), img2_q1)


# Img3

img3 = cv2.imread(os.path.join(source, "img3.jpg"))

img3_q1, img3_labels = paint_image(img3)

watershed_segmentation(img3, img3_q1, img3_labels)
cv2.imwrite(os.path.join(target, "img3_q1.jpg"), img3_q1)