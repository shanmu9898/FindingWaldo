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

## train waldo classifier
feats = []
labels = []

def update_features(feats, labels, path, label, range=None):
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

update_features(feats, labels, WALDO_PATH, 1)
update_features(feats, labels, WALDO_BIG_FACE_PATH, 1)
update_features(feats, labels, OTHER_FACE_PATH, 0, 50)
# update_features(feats, labels, OTHER_NON_FACE_PATH, other_non_face_images[0:13], 0)

svc_linear = SVC(kernel="linear")
svc_linear.fit(feats, labels)

# TODO: replace with pickle save model
test_img = cv2.imread('000.jpg')
test_img = test_img[3000:4500, 8000:]

img_list, coord_list = detect_stripes(test_img)

to_erase = copy.deepcopy(test_img)

preds = []
for img, coord in zip(img_list, coord_list):
    feature, _ = get_hog_feature(image=img)
    prediction = svc_linear.predict([feature])[0]
    preds.append(prediction)
    if prediction == 1:
        print(coord)
        to_erase[coord[0]:coord[0]+128, coord[1]:coord[1]+128] *= 0

plt.imshow(to_erase)
plt.show()

print(len(coord_list), preds)
print(coord_list)

# slide(test_img, 128, 128, svc_linear)