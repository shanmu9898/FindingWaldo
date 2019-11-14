from Training import train
from sliding_window import slideMultiple
from detect_stripes import detect_stripes
import cv2
import os

window_h = 140
window_w = 140
classfiers = train('/Users/Bumblebee/Desktop/Y4S1/CS4243/GroupProjects/FindingWaldo/datasets/faces', window_h,
                   window_w)

print("Starting Prediction")
for filename in os.listdir("/Users/Bumblebee/Desktop/Y4S1/CS4243/GroupProjects/FindingWaldo/datasets/Val"):
    if filename.endswith(".jpg"):
        print(filename)
        filenameCropped = filename.strip(".jpg")
        img = cv2.imread("/Users/Bumblebee/Desktop/Y4S1/CS4243/GroupProjects/CS4243-Project/datasets/JPEGImages/" + filename)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageNeeded = img
        images, coordinates = detect_stripes(imageNeeded)
        print("Length of cordinaates = " + str(len(coordinates)))
        imageBWNeeded = image
        coordinates = slideMultiple(filenameCropped, imageBWNeeded, coordinates, window_h, window_w, classfiers[0])

    else:
        continue
