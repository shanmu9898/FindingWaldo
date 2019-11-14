from detect_stripes import detect_stripes
from cascade_detector import run_cascade
from IOU import IoU
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("C:\\Users\\Lawrence\\Dropbox\\Sem7\\CS4243\\project\\CS4243-Project\\datasets\\JPEGImages\\000.jpg")

# get potential positions of waldo from detect stripes
image_list, stripe_coordinates = detect_stripes(img)

# get predictions of waldo's faces from cascade
cascade_coordinates = run_cascade(img)

# [9602, 3986, 79, 99]
# [9600, 4000, 9800, 4200]

waldo_coordinates = []

for cc in cascade_coordinates:
    # stripe shirts are always in the region below waldo's face

    x, y, w, h = cc
    cascade_img = img[y:y+h, x:x+w]

    for image, coordinate in zip(image_list, stripe_coordinates):

        dy, dx = coordinate
        if IoU(cc, (dx, dy, 200, 200)) > 0:
            print("found")

            waldo_coordinates.append([x, y, 100, 250])


for coord in waldo_coordinates:
    x, y, w, h = coord
    plt.imshow(img[y:y+h, x:x+w])
    plt.show()

