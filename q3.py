import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)


Or_img = cv.imread('Q3.jpg',cv.IMREAD_COLOR)
img = cv.cvtColor(Or_img,cv.COLOR_BGR2Lab)
L,a,b = cv.split(img)
'''cv.imshow("L",L)
cv.imshow("a",a)
cv.imshow("b",b)
cv.waitKey(0)'''

L_new = gammaCorrection(L, 2.2)
gammaImg = cv.merge([L_new,a,b])
fin_img = cv.cvtColor(gammaImg,cv.COLOR_Lab2BGR)

cv.imshow('Original image', Or_img)
cv.imshow('Gamma corrected image', fin_img)
cv.waitKey(0)
cv.destroyAllWindows()

figure, axis = plt.subplots(1, 2)

for i in range(3):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    axis[0].plot(hist)
    axis[0].set_title("Original Image")
    plt.xlim([0, 256])

for i in range(3):
    hist = cv.calcHist([gammaImg], [i], None, [256], [0, 256])
    axis[1].plot(hist)
    axis[1].set_title("Gamma Corrected Image")
    plt.xlim([0, 256])
    
plt.show()
