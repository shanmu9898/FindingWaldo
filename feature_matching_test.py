import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
import os

# sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.KAZE_create()

def get_image_paths(path):
    image_paths = []
    image_names = os.listdir(path)
    
    for image_name in image_names:
        if image_name == '.DS_Store':
            continue
        img_path = os.path.join(path, image_name)
        image_paths.append(img_path)

    return image_paths


def extract_feature_vector(img, vector_size=16):
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


img1 = cv2.imread('./datasets/specs')
feat1 = extract_feature_vector(img1)

images = get_image_paths('./datasets/faces/neg')

count = 0
for image in images:
    img2 = cv2.imread(image)

    feat2 = extract_feature_vector(img2)
    distance = spatial.distance.cosine(feat1, feat2)

    print(f'{image}: {distance}')

    if distance <= 0.45:
        count += 1

print(f'{count}/{len(images)}')