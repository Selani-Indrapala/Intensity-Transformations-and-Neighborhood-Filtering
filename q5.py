import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('shells.tif',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

cv.imshow('Original Image',img)
cv.imshow('New Image',img2)
cv.waitKey(0)
cv.destroyAllWindows()

plt.hist(img.flatten(),256,[0,256], color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('Original Histogram','New Histogram'), loc = 'upper left')
plt.show()