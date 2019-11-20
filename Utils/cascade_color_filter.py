from Utils.detect_stripes import detect_stripes
from Utils.cascade_detector import run_cascade
from Utils.IOU import IoU
import cv2
from Utils.write_file import write_file
import glob
import os


paths = glob.glob(os.path.join(".", "datasets", "Val", "**.jpg"))

for path in paths:
    img = cv2.imread(path)

    img_name = path[-7:-4]

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

    for coord in waldo_coordinates:
        x, y, w, h = coord
        write_file(os.path.join(".", "baseline", "waldo.txt"), img_name, 1, x, y, x+w, y+h)





