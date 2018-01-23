import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

# Python gradient calculation 
 
# Read image
im = cv2.imread('C:/Users/Aashna/Desktop/major project/img.jpg')
im = np.float32(im) / 255.0

#Calculate gradient 
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

# Python Calculate gradient magnitude and direction ( in degrees ) 
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
plt.imshow(mag,'gray')
plt.xticks([]),plt.yticks([])
plt.show()
#cv2.imshow('image',mag)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
