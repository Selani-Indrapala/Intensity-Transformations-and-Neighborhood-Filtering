import cv2 as cv
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

image = cv.imread('einstein.png',cv.IMREAD_GRAYSCALE)

#Filter2D
kernel2 = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
kernel4 = np.array([[1,2,1],[0,0,0], [-1, -2, -1]])
img2 = cv.filter2D(image, -1, kernel2)
img4 = cv.filter2D(image, -1, kernel4)

#Own Code

#Zero padding shape
height, width = image.shape
new_img = np.full((height+2,width+2),0,dtype = np.uint8)
new_img[1:height+1,1:width+1] = image

#Convolution
img_fin = np.zeros(image.shape)
for h in range(height):
    for w in range(width):
        img_fin[h,w] = np.sum(np.multiply(kernel2,new_img[h:h+3,w:w+3]))

#Using the given property
arr1 = np.array([[1],[2],[1]])
arr2 = np.array([[1,0,-1]])
img1 = sig.convolve2d(image,arr1,mode="same")
img_prop = sig.convolve2d(img1,arr2,mode="same")

#Output Images
fig,ax = plt.subplots(1,4)
ax[0].imshow(image, cmap = 'gray')
ax[0].set_title("Original Image")
ax[1].imshow(img2, cmap = 'gray')
ax[1].set_title("Filter2D")
ax[2].imshow(img_fin, cmap = 'gray')
ax[2].set_title("Own code")
ax[3].imshow(img_prop, cmap = 'gray')
ax[3].set_title("Given Property")
for i in range(4):
    ax[i].set_xticks([]), ax[i].set_yticks([])
plt.show()
