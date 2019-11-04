import os
import numpy as np
from sklearn.svm import SVC

from feature_extraction import get_hog_feature
from sliding_window import slide # TODO: remove

####################
WALDO_PATH = os.path.join('.', 'faces', 'waldo')
WENDA_PATH = os.path.join('.', 'faces', 'wenda')
WIZARD_PATH = os.path.join('.', 'faces', 'wizard')
OTHER_FACE_PATH = os.path.join('.', 'faces', 'other_face')
OTHER_NON_FACE_PATH = os.path.join('.', 'faces', 'non_face')
####################

waldo_images = os.listdir(WALDO_PATH)
wenda_images = os.listdir(WALDO_PATH)
wizard_images = os.listdir(WIZARD_PATH)
other_face_images = os.listdir(OTHER_FACE_PATH)
other_non_face_images = os.listdir(OTHER_NON_FACE_PATH)

## train waldo classifier
feats = []
labels = []

def update_features(feats, labels, path, images, label):
    for image in images:
        if image == './DS_Store': # Mac burden
            continue
        img_path = f'{path}/{image}'
        fd, _ = get_hog_feature(img_path)
        feats.append(fd)
        labels.append(label)

update_features(feats, labels, WALDO_PATH, waldo_images, 1)
update_features(feats, labels, OTHER_FACE_PATH, other_face_images, 0)
update_features(feats, labels, OTHER_NON_FACE_PATH, other_non_face_images, 0)

svc_linear = SVC(kernel="linear")
svc_linear.fit(feats, labels)

# TODO: replace with pickle save model
test_img = cv2.imread('')
slide(test_img, 128, 128, svc_linear)