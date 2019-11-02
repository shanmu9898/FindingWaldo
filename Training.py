from __future__ import print_function
from imutils import paths
from scipy import io
import numpy as np
import random
import cv2
import pickle as cPickle
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import data, exposure
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

def train(path, height, width):
    data = []
    labels = []
    waldo_paths = list(paths.list_images(path + "/waldo"))
    wenda_paths = list(paths.list_images(path + "/wenda"))
    wizard_paths = list(paths.list_images(path + "/wizard"))
    neg_paths = list(paths.list_images(path + "/neg"))
    print("Processing Waldo for training")
    for (i, waldo_path) in enumerate(waldo_paths):  # load training image
        image = cv2.imread(waldo_path)
        # convert to grayscale
        train_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize
        train_image = cv2.resize(train_image, (width, height), interpolation=cv2.INTER_AREA)
        # put the train_image into a list called train_image_list
        train_image_list = (train_image, cv2.flip(train_image, 0),cv2.flip(train_image, -1), cv2.flip(train_image, 1))

        # loop for train_image in train_image_list
        for train_image in train_image_list:
            # extract features from train_image and add to list of features
            features, hog_image = hog(train_image, orientations=18, pixels_per_cell=(16, 16), cells_per_block=(3, 3),
                                      visualize=True, multichannel=False)

            data.append(features)
            labels.append(1)

    print("Processing Wenda for training")
    for (i, wenda_path) in enumerate(wenda_paths):  # load training image
        image = cv2.imread(wenda_path)
        # convert to grayscale
        train_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize
        train_image = cv2.resize(train_image, (width, height), interpolation=cv2.INTER_AREA)
        # put the train_image into a list called train_image_list
        train_image_list = (train_image, cv2.flip(train_image, 0), cv2.flip(train_image, -1), cv2.flip(train_image, 1))

        # loop for train_image in train_image_list
        for train_image in train_image_list:
            # extract features from train_image and add to list of features
            features, hog_image = hog(train_image, orientations=18, pixels_per_cell=(16, 16), cells_per_block=(3, 3),
                                      visualize=True, multichannel=False)

            data.append(features)
            labels.append(2)

    print("Processing Wizard for training")
    for (i, wizard_path) in enumerate(wizard_paths):  # load training image
        image = cv2.imread(wizard_path)
        # convert to grayscale
        train_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize
        train_image = cv2.resize(train_image, (width, height), interpolation=cv2.INTER_AREA)
        # put the train_image into a list called train_image_list
        train_image_list = (train_image, cv2.flip(train_image, 0), cv2.flip(train_image, -1), cv2.flip(train_image, 1))

        # loop for train_image in train_image_list
        for train_image in train_image_list:
            # extract features from train_image and add to list of features
            features, hog_image = hog(train_image, orientations=18, pixels_per_cell=(16, 16), cells_per_block=(3, 3),
                                      visualize=True, multichannel=False)
            data.append(features)
            labels.append(3)

    print("Processing Negative examples for training")
    for (i, neg_path) in enumerate(neg_paths):  # load training image
        image = cv2.imread(neg_path)
        # convert to grayscale
        train_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize
        train_image = cv2.resize(train_image, (width, height), interpolation=cv2.INTER_AREA)
        # put the train_image into a list called train_image_list
        train_image_list = (train_image, cv2.flip(train_image, 0), cv2.flip(train_image, -1), cv2.flip(train_image, 1))

        # loop for train_image in train_image_list
        for train_image in train_image_list:
            # extract features from train_image and add to list of features
            features, hog_image = hog(train_image, orientations=18, pixels_per_cell=(16, 16), cells_per_block=(3, 3),
                                      visualize=True, multichannel=False)
            data.append(features)
            labels.append(4)

    print("1/2: Training classifiers...")
    print("SVC classifier started")
    modelSVCLinear = SVC(kernel="linear", probability=True, random_state=22)


    print("Decision Tree Classifier started")
    modelDecisiontree = DecisionTreeClassifier(max_depth=2)

    print("Random Forest Classifier started")
    # Create the model with 100 trees
    modelRandomForest = RandomForestClassifier(n_estimators=30, bootstrap=True, max_features='sqrt')


    print("Printing CrossValidation Scores of each of them")
    # kfold = KFold(n_splits=5, random_state=13, shuffle=True)
    # accuracyScoresSVC = cross_val_score(modelSVCLinear, data, labels, cv=kfold, scoring = 'f1_macro')
    # print("Accuracy Cross Validtion scores for SVC are ")
    # print(accuracyScoresSVC)
    # accuracyScoresDT = cross_val_score(modelDecisiontree, data, labels, cv=kfold, scoring = 'f1_macro')
    # print("Accuracy Cross Validtion scores for DT are ")
    # print(accuracyScoresDT)
    # accuracyScoresRF = cross_val_score(modelRandomForest, data, labels, cv=kfold, scoring = 'f1_macro')
    # print("Accuracy Cross Validtion scores for RF are ")
    # print(accuracyScoresRF)

    print("Shuffling being done before final fitting classifiers on all data")
    # data, labels = shuffle(data, labels)
    print("Finally fitting being done for all the classifiers")
    print("SVC being fit on all data, labels")
    modelSVCLinear.fit(data, labels)
    print("DT being fit on all data, labels")
    modelDecisiontree.fit(data,labels)
    print("RF being fit on all data, labels")
    modelRandomForest.fit(data, labels)

    print("Creating a classifier list")
    classifiers = [modelSVCLinear, modelDecisiontree, modelRandomForest]

    print("Done!")
    return classifiers

if __name__ == "__main__":
   classifiers = train("/Users/Bumblebee/Desktop/Y4S1/CS4243/GroupProjects/FindingWaldo/datasets/faces", 90, 60)
   print("FINAL DONE")

