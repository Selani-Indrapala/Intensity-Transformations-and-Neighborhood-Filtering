import numpy as np
import cv2
import math

def GetBilinearPixel(imArr, posX, posY):
        out = []
 
      
        modXi = int(posX)
        modYi = int(posY)
        modXf = posX - modXi
        modYf = posY - modYi
        modXiPlusOneLim = min(modXi+1,imArr.shape[1]-1)
        modYiPlusOneLim = min(modYi+1,imArr.shape[0]-1)
 
      
        for chan in range(imArr.shape[2]):
                bl = imArr[modYi, modXi, chan]
                br = imArr[modYi, modXiPlusOneLim, chan]
                tl = imArr[modYiPlusOneLim, modXi, chan]
                tr = imArr[modYiPlusOneLim, modXiPlusOneLim, chan]
 
               
                b = modXf * br + (1. - modXf) * bl
                t = modXf * tr + (1. - modXf) * tl
                pxf = modYf * t + (1. - modYf) * b
                out.append(int(pxf+0.5))
       
        return out

img = cv2.imread('zooming\zooming\im01small.png',1)
width,height,chan = img.shape


scale_x=2
scale_y=2

#w2= 870
#h2=870
w1=width
h1=height
w2=int(math.floor(w1*scale_x))
h2=int(math.floor(h1*scale_y))
img_bl = np.empty((w2,h2,chan), dtype=np.uint8)
#x_ratio=float(w1/float(w2))
#y_ratio=float(h1/float(h2))
x_ratio=float(1/float(scale_x))
y_ratio=float(1/float(scale_y))

for i in range(0,w2):
    for j in range(0,h2):
        orir = i * x_ratio #Find position in original image
        oric = j * y_ratio
        
        img_bl[i, j]= GetBilinearPixel(img, oric, orir)

cv2.imshow("Original Image",img)
cv2.imshow("Scaled Image", img_bl)              
cv2.waitKey(0)
cv2.destroyAllWindows()