import cv2
import numpy as np

img = cv2.imread("resources/sample.png")
print(img.shape)

imgResize = cv2.resize(img, (100,100))
print(imgResize.shape)

imgCropped = img[0:200,50:200]

cv2.imshow("image",img)
cv2.imshow("resized",imgResize)
cv2.imshow("cropped",imgCropped)
cv2.waitKey(0)