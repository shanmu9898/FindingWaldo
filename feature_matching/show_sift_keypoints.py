import cv2
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Show top n SIFT keypoints with the highest response')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to image')
parser.add_argument('-n', action='store', 
                    type=int, required=True, help='n')

args = vars(parser.parse_args())
path = args['p']
n = args['n']

img = cv2.imread(path, 0)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(img, None)
kp = sorted(kp, key=lambda x: -x.response)[:n]
img = cv2.drawKeypoints(img, kp, outImage=None)
plt.imshow(img), plt.show()