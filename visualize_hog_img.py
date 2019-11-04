import argparse
import os
import cv2
from pathlib import Path

from feature_extraction import get_hog_feature

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
        img_name = f'./test/{Path(image_path).stem}_hog.png'

        cv2.imwrite(img_name, hog_img)
