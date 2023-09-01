import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv.imread('daisy.jpg')
height, width, _ = img.shape
left_margin_proportion = 0.1 
right_margin_proportion = 0.1 
up_margin_proportion = 0.1 
down_margin_proportion = 0.45

rect = (
    int(width * left_margin_proportion),
    int(height * up_margin_proportion),
    int(width * (1 - right_margin_proportion)),
    int(height * (1 - down_margin_proportion)),
)
# Create a 0's mask
mask = np.zeros(img.shape[:2],np.uint8)
# Create 2 arrays for background and foreground model
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

#rect = (20,199,170,170)
mask, bgdModel, fgdModel = cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

main_mask = (mask == cv.GC_PR_FGD).astype("uint8")*255
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_fg = img*mask2[:,:,np.newaxis]

mask3 = np.where((mask==3)|(mask==1),0,1).astype('uint8')
img_bg = img*mask3[:,:,np.newaxis]

cv.imshow("Foreground",img_fg)
cv.imshow("Background",img_bg)
cv.imshow("Original Image",img)
cv.imshow("Mask",main_mask)
cv.waitKey(0)
cv.destroyAllWindows()

#Blurring Background
bg_blurred = cv.blur(src=img_bg, ksize=(10, 10))
cv.imshow('Blurred Background', bg_blurred)
cv.waitKey(0)
cv.destroyAllWindows()

#Adding background and foreground together
fin_img = cv.add(bg_blurred,img_fg)
cv.imshow("Original Image",img)
cv.imshow('Final Image',fin_img)
cv.waitKey(0)
cv.destroyAllWindows()
