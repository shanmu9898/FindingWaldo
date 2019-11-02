from Training import train
from sliding_window import slide
import glob
import cv2

paths = glob.glob('C:\\Users\\Lawrence\\Dropbox\\Sem7\\CS4243\\project\\CS4243-Project\\datasets\\JPEGImages\\**.jpg')
window_h = 120
window_w = 100

for path in paths[:1]:

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    classfiers = train('.\\datasets\\faces')

    coordinates = slide(img, window_h, window_w, classfiers[0])

    print(coordinates)
