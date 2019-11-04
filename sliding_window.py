import cv2
import numpy as np
import matplotlib.pyplot as plt

# for hog-svm
from skimage.feature import hog
import copy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from feature_extraction import get_hog_feature

def slide(img, window_h, window_w, classifier):

    height, width = img.shape[0], img.shape[1]

    toErase = copy.deepcopy(img)

    step_size = 20

    coordinates = []

    for i in range(0, height - window_h, step_size):
        if(i % 100 == 0):
            print(i)
        for j in range(0, width - window_w, step_size):
            window = img[i:i+window_h, j:j+window_w]

            # features, hog_image = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
            #                           visualize=True, multichannel=False)

            features, _ = get_hog_feature(image=window)

            prediction = classifier.predict([features])[0]

            if prediction == 1:
                toErase[i:i+window_h, j:j+window_w] *= 0

            else:
                coordinates.append([i, j])

    plt.imshow(toErase)
    plt.show()

    return coordinates


def slideMultiple(img, coordinates, window_h, window_w, classifier):
    toErase = copy.deepcopy(img)
    counter = 1
    for coordinate in coordinates:
        print("Cordinate number " + str(counter))
        counter = counter + 1
        height, width = 250, 250

        step_size = 20
        print("Coordinate for loop is " + str(coordinate))
        x,y = coordinate[0], coordinate[1]

        for i in range(x, x + height - window_h, step_size):
            if(i % 100 == 0):
                print(i)
            for j in range(y, y + width - window_w, step_size):
                window = img[i:i+window_h, j:j+window_w]

                features, hog_image = hog(window, orientations=18, pixels_per_cell=(16, 16), cells_per_block=(3, 3),
                                          visualize=True, multichannel=False)

                prediction = classifier.predict([features])[0]

                if prediction == 1:
                    toErase[i:i+window_h, j:j+window_w] *= 0

    plt.imshow(toErase)
    plt.show()






