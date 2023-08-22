import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def IntensityTransform(x,alpha,sigma):
    f1 = x + (alpha/128)*np.exp(-np.power(x-128,2)/(2*np.power(sigma,2)))
    f2 = 255
    f1[f1>f2]=f2
    return f1


img = cv.imread('spider.png',cv.IMREAD_COLOR)
img = cv.cvtColor(img,cv.COLOR_BGR2HSV)

h,s,v = cv.split(img)
cv.imshow("Hue",h)
cv.imshow("Saturation",s)
cv.imshow("value",v)
cv.waitKey(0)

transImage = IntensityTransform(s,1,70)
cv.imshow("Transformed Saturation",transImage)
cv.waitKey(0)
cv.destroyAllWindows()