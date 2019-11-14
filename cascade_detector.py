import cv2
import os
import matplotlib.pyplot as plt


# requires img in grayscale
# returns coordinates of window in the format x, y, w, h
def run_cascade(img):

    cascade_path = os.path.join(".", "cascade", "cascade.xml")

    waldo_cascade = cv2.CascadeClassifier(cascade_path)

    img = cv2.imread("C:\\Users\\Lawrence\\Dropbox\\Sem7\\CS4243\\project\\CS4243-Project\\datasets\\JPEGImages\\000.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    count = 0

    waldo = waldo_cascade.detectMultiScale(gray, 1.1, 3)
    coordinates = []
    for x,y,w,h in waldo:
        if w < 200 and h < 200:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            coordinates.append([x, y, w, h])
            count += 1

    print("Number of windows found", count)
    plt.imshow(img)
    plt.show()

    return coordinates
