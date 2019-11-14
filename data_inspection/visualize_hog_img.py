import argparse
import os
import cv2
from pathlib import Path
from skimage.feature import hog

def get_hog_feature(path=None, image=None, img_height=128, img_width=128, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), multichannel=True):

    if path is not None:
        img = cv2.imread(path)

        print(path)

        if not multichannel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = image

    resized_img = cv2.resize(img, (img_width, img_height))

    fd, hog_image = hog(resized_img, orientations=orientations, pixels_per_cell=pixels_per_cell, 
                    cells_per_block=cells_per_block, visualize=True, multichannel=multichannel)

    return fd, hog_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise hog images.')
    parser.add_argument('-p', action='store', 
                        type=str, required=True, help='Path to directories that contains all images')
    
    args = vars(parser.parse_args())
    path = args['p']

    image_paths = os.listdir(path)

    for image_path in image_paths:
        if image_path == '.DS_Store': # Mac burden
            continue

        _, hog_img = get_hog_feature(f'{path}/{image_path}')
        img_name = f'./hog/{Path(image_path).stem}_hog.png'
        cv2.imwrite(img_name, hog_img)
