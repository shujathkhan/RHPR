import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

x=0
for image in glob.glob('C:/Users/Aashna/Desktop/major project/sitting/*.jpg'):
    x=x+1
    img=cv2.imread(image,0)

#edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    img = cv2.medianBlur(img,5)

#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Adaptive Gaussian Thresholding']
    images = [img, th3]
    for i in range(0,2):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
        cv2.imwrite('C:/Users/Aashna/Desktop/major project/output/sitting/{}.jpg'.format(x),th3)
plt.show() 

    

