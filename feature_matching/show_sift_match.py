import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
from scipy import spatial

parser = argparse.ArgumentParser(description='Show SIFT feature matching')
parser.add_argument('-p1', action='store', 
                    type=str, required=True, help='Path to image 1')
parser.add_argument('-p2', action='store', 
                    type=str, required=True, help='Path to image 2')

args = vars(parser.parse_args())
path1 = args['p1']
path2 = args['p2']

img1 = cv2.imread(path1, 0)
img2 = cv2.imread(path2, 0)

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
my_dist = 0
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        print(spatial.distance.cosine(des1[m.queryIdx], des2[m.trainIdx]))
        my_dist += spatial.distance.cosine(des1[m.queryIdx], des2[m.trainIdx])

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=None,flags=2)
plt.imshow(img3), plt.show()

print(my_dist/len(good))