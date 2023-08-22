import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

c = np.array([(50,50),(50,100),(150,255),(150,150),(255,255)])

t1 = np.linspace(0,c[0,1],c[0,0]+1-0).astype('uint8')
t2 = np.linspace(c[0,1]+1,c[1,1],c[1,0]-c[0,0]).astype('uint8')
t3 = np.linspace(c[1,1]+1,c[2,1],c[2,0]-c[1,0]).astype('uint8')
t4 = np.linspace(c[2,1]+1,c[3,1],c[3,0]-c[2,0]).astype('uint8')
t5 = np.linspace(c[3,1]+1,c[4,1],c[4,0]-c[3,0]).astype('uint8')
t6 = np.linspace(c[4,1]+1,255,255-c[4,0]).astype('uint8')

transform = np.concatenate((t1,t2),axis=0).astype('uint8')
transform = np.concatenate((transform,t3),axis=0).astype('uint8')
transform = np.concatenate((transform,t4),axis=0).astype('uint8')
transform = np.concatenate((transform,t5),axis=0).astype('uint8')
transform = np.concatenate((transform,t6),axis=0).astype('uint8')

fig,ax = plt.subplots()
ax.plot(transform)
ax.set_xlabel(r'Input')
ax.set_ylabel(r'Output')
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_aspect('equal')
plt.show()

img_orig = cv.imread('emma.jpg',cv.IMREAD_GRAYSCALE)
cv.namedWindow('Image',cv.WINDOW_AUTOSIZE)
cv.imshow('Image',img_orig)
cv.waitKey(0)
image_transformed = cv.LUT(img_orig,transform)
cv.imshow('Transformed Image',image_transformed)
cv.waitKey(0)
cv.destroyAllWindows()