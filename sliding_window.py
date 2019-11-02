import cv2
import numpy as np
import matplotlib.pyplot as plt

# for hog-svm
from skimage.feature import hog
import copy
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def slide(img, window_h, window_w, classifier):

    height, width = img.shape[0], img.shape[1]

    toErase = copy.deepcopy(img)

    step_size = 20

    coordinates = []

    for i in range(0, height - window_h, step_size):
        for j in range(0, width - window_w, step_size):
            window = img[i:i+window_h, j:j+window_w]

            features, hog_image = hog(window, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                                      visualize=True, multichannel=False)

            prediction = classifier.predict([hog_image])

            if prediction != "waldo" or prediction != "wenda" or prediction != "wizard":
                toErase[i:i+window_h, j:j+window_w] *= 0

            else:
                coordinates.append([i, j])

    plt.imshow(toErase)
    plt.show()

    return coordinates






