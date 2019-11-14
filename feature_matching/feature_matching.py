import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
import os
import argparse

sift = cv2.xfeatures2d.SIFT_create()

def extract_feature_vector(img, top_n=20):
    try:
        keypoints = sift.detect(img)
        keypoints = sorted(keypoints, key=lambda x: -x.response)[:20]

        keypoints, descriptors = sift.compute(img, keypoints)

        if descriptors is None: # No keypoints
            descriptors = np.array([])

        feature_vector = descriptors.flatten()

        needed_size = (top_n * 128)
        if feature_vector.size < needed_size:
            feature_vector = np.concatenate([feature_vector, np.zeros(needed_size - feature_vector.size)])

    except cv2.error as e:
        print(f'Error: {e}')
        return None

    return feature_vector

parser = argparse.ArgumentParser(description='SIFT feature matching')
parser.add_argument('-t', action='store', 
                    type=str, required=True, help='Template img path')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Test img path')

args = vars(parser.parse_args())
template_img_path = args['t']
test_img_path = args['p']

template_img = cv2.imread(template_img_path, 0)
template_feat = extract_feature_vector(template_img)

test_img = cv2.imread(test_img_path, 0)

# can modify these
scales = [1, 2, 4]
window_height = 50
window_width = 40

test_height, test_width = test_img.shape
step = 10
threshold = 0.5

taken = []
for h in range(0, test_height, step):
    for w in range(0, test_width, step):
        for scale in scales:
            wh = int(scale * window_height)
            ww = int(scale * window_width)

            if h+wh >= test_height or w+ww >= test_width:
                continue
            
            test_segment = test_img[h:h+wh, w:w+ww]
            test_feat = extract_feature_vector(test_segment)
            
            distance = spatial.distance.cosine(template_feat, test_feat)
            print(distance)
            
            if distance <= threshold:
                # top right hand corner x, y, window_width, window_height
                data = (w, h, ww, wh)
                taken.append(data)
                cv2.imwrite(f'h{h}w{w}.jpg', test_segment)
                break

print(f'{len(data)} objects detected')