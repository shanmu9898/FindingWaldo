# feature matching using SIFT features
# variable sliding window size
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial

# Initialise SIFT detector
sift = cv2.KAZE_create()
# sift = cv2.xfeatures2d.SIFT_create()

# helper functions
def extract_feature_vector(img, vector_size=64):
    try:
        keypoints = sift.detect(img)
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]

        keypoints, descriptors = sift.compute(img, keypoints)

        feature_vector = descriptors.flatten()

        needed_size = (vector_size * 64)
        if feature_vector.size < needed_size:
            feature_vector = np.concatenate([feature_vector, np.zeros(needed_size - feature_vector.size)])

    except cv2.error as e:
        print(f'Error: {e}')
        return None

    return feature_vector

# load the train image
### CAN MODIFY THIS!! ###
train_img = cv2.imread('./datasets/faces/waldo/026_1_0.jpg')
test_img = cv2.imread('./datasets/faces/waldo/000_0_0.jpg')
#########################

train_feat = extract_feature_vector(train_img)

# sliding window scale and its scale at every point
### CAN MODIFY THIS!! ###
scales = [0.1, 0.25, 0.5, 1, 1.5, 2]
window_height = 128
window_width = 128
#########################

# sliding window begins here
test_height, test_width, _ = test_img.shape
step = 5 ### CAN MODIFY THIS!! ###
threshold = 0.45 ### CAN MODIFY THIS!!

taken = []
for h in range(0, test_height, step):
    for w in range(0, test_width, step):
        for scale in scales:
            print(h, w, scale)
            wh = int(scale * window_height)
            ww = int(scale * window_width)

            if h+wh >= test_height or w+ww >= test_width:
                continue
            
            test_segment = test_img[h:h+wh, w:w+ww]
            test_feat = extract_feature_vector(test_segment)
            
            distance = spatial.distance.cdist(train_feat, test_feat, 'cosine')
            # distance = spatial.distance.cosine(train_feat, test_feat)
            print(distance)
            
            if distance <= threshold:
                # top right hand corner x, y, window_width, window_height
                data = (w, h, ww, wh)
                taken.append(data)
                cv2.imwrite(f'{h}{w}.jpg', test_segment)
                break

# show image or save data, etc.
