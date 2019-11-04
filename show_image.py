
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('000.jpg')
img = img[3000:4500, 8000:]
plt.imshow(img)
plt.show()