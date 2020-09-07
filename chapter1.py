import cv2
import numpy as np

img = cv2.imread("resources/sample.png")
kernel = np.ones((5, 5), np.uint8)

#changing something

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(img, 150, 200)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)
imgEroded = cv2.erode(imgCanny, kernel, iterations=1)

cv2.imshow("grayscale", imgGray)
cv2.imshow("blurred", imgBlur)
cv2.imshow("edges", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Erosion Image", imgEroded)
cv2.waitKey(0)