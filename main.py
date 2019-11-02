from Training import train
from sliding_window import slideMultiple
from detect_stripes import detect_stripes
import cv2

window_h = 120
window_w = 100

img = cv2.imread("/Users/Bumblebee/Desktop/Y4S1/CS4243/GroupProjects/CS4243-Project/datasets/JPEGImages/007.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
classfiers = train('/Users/Bumblebee/Desktop/Y4S1/CS4243/GroupProjects/FindingWaldo/datasets/faces', window_h, window_w)
imageNeeded = img
coordinates = detect_stripes(imageNeeded)
print("Length of cordinaates = " + str(len(coordinates)))
print("Co ordinates are " + str(coordinates))
imageBWNeeded = image
coordinates = slideMultiple(imageBWNeeded, coordinates, window_h, window_w, classfiers[0])


print(coordinates)
