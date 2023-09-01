import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('jeniffer.jpg',cv.IMREAD_COLOR)

# resize image
scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height) 
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
gr = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#Splitting into HSV planes
h,s,v = cv.split(hsv_img)


cv.imshow("Hue",h)
cv.imshow("Saturation",s) #Choose Saturation Plane
cv.imshow("value",v)
cv.waitKey(0)
cv.destroyAllWindows()

#Getting the mask
ret, mask1 = cv.threshold(s, 12, 255, cv.THRESH_BINARY)


cv.imshow('Mask',mask1)
cv.waitKey(0)
cv.destroyAllWindows()

#Getting the foreground
foreground = cv.bitwise_and(gr,gr,mask=mask1)


cv.imshow('Foreground',foreground)
cv.waitKey(0)
cv.destroyAllWindows()

#Getting the cdf
hist,bins = np.histogram(foreground.ravel(),256,[0,256])
cdf = hist.cumsum()

#Equilising the foreground
cdf_normalized = cdf * hist.max()/ cdf.max()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
equ = cdf[foreground]


cv.imshow('Equalised Foreground',equ)
cv.waitKey(0)
cv.destroyAllWindows()

#Plotting the histograms
figure, axis = plt.subplots(1, 2)
axis[0].hist(foreground.flatten())
axis[0].set_title("Original Histogram")
axis[1].hist(equ.flatten())
axis[1].set_title("Equalised Histogram")
plt.show()

#Extracting the background
mask_inv = cv.bitwise_not(mask1)
background = cv.bitwise_and(gr,gr,mask=mask_inv)

cv.imshow('Background',background)
cv.waitKey(0)
cv.destroyAllWindows()

#Adding both background and equalised foreground
fin_img = cv.add(background,equ)

cv.imshow("Original Image",gr)
cv.imshow('Final Image',fin_img)
cv.waitKey(0)
cv.destroyAllWindows()
