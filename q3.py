import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)


img = cv.imread('Q3.jpg',cv.IMREAD_COLOR)
img = cv.cvtColor(img,cv.COLOR_BGR2Lab)
gammaImg = gammaCorrection(img, 2.2)

cv.imshow('Original image', img)
cv.imshow('Gamma corrected image', gammaImg)
cv.waitKey(0)
cv.destroyAllWindows()

figure, axis = plt.subplots(1, 2)

for i, col in enumerate(['b', 'g', 'r']):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    axis[0].plot(hist, color = col)
    axis[0].set_title("Original Image")
    plt.xlim([0, 256])

for i, col in enumerate(['b', 'g', 'r']):
    hist = cv.calcHist([gammaImg], [i], None, [256], [0, 256])
    axis[1].plot(hist, color = col)
    axis[1].set_title("Gamma Corrected Image")
    plt.xlim([0, 256])
    
plt.show()