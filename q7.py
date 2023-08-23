import cv2 as cv
import numpy as np

image = cv.imread('einstein.png',cv.IMREAD_COLOR)

kernel2 = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
kernel4 = np.array([[1,2,1],[0,0,0], [-1, -2, -1]])

img2 = cv.filter2D(image, -1, kernel2)
img4 = cv.filter2D(image, -1, kernel4)

#Applying Filter2D
cv.imshow('Original', image)
cv.imshow('Filter2D', img2)
cv.imshow('Given Property', img4)
cv.waitKey(0)
cv.destroyAllWindows()
