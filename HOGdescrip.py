import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
args = vars(ap.parse_args())

# Read image
img = cv2.imread(args["image"])
img = np.float32(img) / 255.0

# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

while True:
    plt.subplot(121), plt.imshow(gx, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gy, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    k = cv2.waitKey(27) & 0xff
    if k == 27:
        break
