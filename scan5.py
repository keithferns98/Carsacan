import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('view1.jpeg')
cv2.imshow("ogimg",img)
width=900
height=1100
dimension=(width,height)
resized=cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)
print(resized.shape)
cv2.imshow('output',resized)
#save the image
cv2.imwrite('resized_img.jpg',resized)
gaussain=cv2.GaussianBlur(img,(5,5),0)
gaussain2=cv2.GaussianBlur(resized,(5,5),0)
cv2.imshow('denoiseimg',gaussain)
cv2.imshow('denoiseimg',gaussain2)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (1250,480,3250,1800)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]


plt.imshow(img)
plt.colorbar()
plt.show()
