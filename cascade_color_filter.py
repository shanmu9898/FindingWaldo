from detect_stripes import detect_stripes
from cascade_detector import run_cascade
from IOU import IoU
import cv2
import matplotlib.pyplot as plt
import copy


img = cv2.imread("C:\\Users\\Lawrence\\Dropbox\\Sem7\\CS4243\\project\\CS4243-Project\\datasets\\JPEGImages\\000.jpg")

height, width = img.shape[0], img.shape[1]

# get potential positions of waldo from detect stripes
image_list, stripe_coordinates = detect_stripes(img)

# get predictions of waldo's faces from cascade
cascade_coordinates = run_cascade(img)

waldo_coordinates = []

for cc in cascade_coordinates:
    # stripe shirts are always in the region below waldo's face

    x, y, w, h = cc
    cascade_img = img[y:y+h, x:x+w]

    for image, coordinate in zip(image_list, stripe_coordinates):

        dy, dx = coordinate
        if IoU(cc, (dx, dy, 200, 200)) > 0:
            print("found")

            if x + 100 >= width or y + 250 >= height:
                continue

            waldo_coordinates.append([x, y, 100, 250])


waldo_final_coordinates = copy.deepcopy(waldo_coordinates)
# remove duplicate coordinates
for i in range(len(waldo_coordinates) - 1):

    coord = waldo_coordinates[i]
    is_dup = False

    for j in range(i+1, len(waldo_coordinates)):

        coord2 = waldo_coordinates[j]

        if IoU(coord, coord2) > 0:
            waldo_final_coordinates.remove(coord)


for coord in waldo_final_coordinates:
    x, y, w, h = coord
    plt.imshow(img[y:y+h, x:x+w])
    plt.show()

