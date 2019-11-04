import os
import numpy as np
from sklearn.svm import SVC
import cv2
import copy
import matplotlib.pyplot as plt

from feature_extraction import get_hog_feature
from sliding_window import slide # TODO: remove
from detect_stripes import detect_stripes # TODO: remove

####################
WALDO_PATH = os.path.join('.', 'datasets', 'faces', 'waldo')
WENDA_PATH = os.path.join('.', 'datasets', 'faces', 'wenda')
WIZARD_PATH = os.path.join('.', 'datasets', 'faces', 'wizard')
OTHER_FACE_PATH = os.path.join('.', 'datasets', 'faces', 'other_face')
OTHER_NON_FACE_PATH = os.path.join('.', 'datasets', 'faces', 'non_face')
WALDO_BIG_FACE_PATH = os.path.join('.', 'datasets', 'faces', 'waldo_big_face')
####################

## train classifier
feats = []
labels = []

def update_features(path, label, range=None):
    images = os.listdir(path)

    if range is not None:
        images = images[:range]

    for image in images:
        if image == '.DS_Store': # Mac burden
            continue
        img_path = f'{path}/{image}'
        fd, _ = get_hog_feature(img_path)
        feats.append(fd)
        labels.append(label)

### Change the following code to change the data used to train the classifier
update_features(WALDO_PATH, 1)
update_features(WALDO_BIG_FACE_PATH, 1)
update_features(OTHER_FACE_PATH, 0, 50)
update_features(OTHER_NON_FACE_PATH, 0, 45)
#############################################################################

svc_linear = SVC(kernel="linear")
svc_linear.fit(feats, labels)

# TODO: replace with pickle save model
test_img = cv2.imread('007.jpg')
# test_img = test_img[3000:4500, 8000:]

img_list, coord_list = detect_stripes(test_img)

to_erase = copy.deepcopy(test_img)
window_h, window_w = 128, 128
for img, coord in zip(img_list, coord_list):
    H, W, _ = img.shape
    step_size = 20
    for h in range(0, H - window_h, step_size):
        for w in range(0, W- window_w, step_size):
            window = img[h:h+window_h, w:w+window_w]
            feature, _ = get_hog_feature(image=window)
            prediction = svc_linear.predict([feature])[0]

            if prediction == 1:
                print(coord, h, w)
                actual_h = h+coord[0]
                actual_w = w+coord[1]
                to_erase[actual_h:actual_h+window_h, actual_w:actual_w+window_w] *= 0

plt.imshow(to_erase)
plt.show()
