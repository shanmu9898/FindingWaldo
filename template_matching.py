# feature matching using SIFT features
# variable sliding window size
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial

# Initialise SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# helper functions
def extract_feature_vector(img, vector_size=64):
    try:
        keypoints = sift.detect(img)

        # Getting the first k features; k=vector_size
        # Number of keypoints varies depending on the image size and color pallet
        # Sorting keypoints based on keypoint response value, the bigger the better
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:vector_size]
        
        # Compute descriptors vectors
        keypoints, descriptors = sift.compute(img, keypoints)

        if descriptors is None:
            descriptors = np.array([])
        # Flatten all of them in one big vector
        # This is our feature vector
        feature_vector = descriptors.flatten()

        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)

        if feature_vector.size < needed_size:
            # concatenate with zeros
            feature_vector = np.concatenate([feature_vector, np.zeros(needed_size - feature_vector.size)])

        if feature_vector.size > needed_size:
            feature_vector = feature_vector[:needed_size]

    except cv2.error as e:
        print(f'Error: {e}')
        return None

    return feature_vector

# load the train image
### CAN MODIFY THIS!! ###
train_img = cv2.imread('008_0_0.jpg')
test_img = cv2.imread('000.jpg')
#########################

train_feat = extract_feature_vector(train_img)

# sliding window scale and its scale at every point
### CAN MODIFY THIS!! ###
scales = [0.5, 1, 2]
window_height = 128
window_width = 128
#########################

# sliding window begins here
test_height, test_width, _ = test_img.shape
step = 50 ### CAN MODIFY THIS!! ###
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
            
            distance = spatial.distance.cosine(train_feat, test_feat)
            print(distance)
            if distance <= threshold:
                # top right hand corner x, y, window_width, window_height
                data = (w, h, ww, wh)
                taken.append(data)
                cv2.imwrite(f'{h}{w}.jpg', test_segment)
                break

# show image or save data, etc.
