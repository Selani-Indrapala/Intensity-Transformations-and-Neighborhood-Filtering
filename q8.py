import numpy as np
import cv2
import math
angle=45
radians = float(angle*(math.pi/180))
img = cv2.imread('zooming\zooming\im01small.png',1)
width,height,chan = img.shape


#scale_x=2
#scale_y=2

w2= 870
h2=870
w1=width
h1=height
img_nn = np.empty((w2,h2,chan), dtype=np.uint8)
x_ratio=float(w1/float(w2))
y_ratio=float(h1/float(h2))
#x_ratio=float(1/float(scale_x))
#y_ratio=float(1/float(scale_y))

for i in range(0,w2):
    for j in range(0,h2):
        p_x=math.floor(j*x_ratio)
        p_y=math.floor(i*y_ratio)
      
        img_nn[j,i]=img[int(p_x),int(p_y)]
           
cv2.imshow("Original Image",img)
cv2.imshow("Scaled Image", img_nn)              
cv2.waitKey(0)
cv2.destroyAllWindows()