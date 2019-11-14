import os
import pandas as pd
import argparse
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot graph of height and weight')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to images')

args = vars(parser.parse_args())
path = args['p']

img_dim_data = []
height = [] # y_axis of scatter plot
width = [] # x_axis of scatter plot
images = os.listdir(path)

for image in images:
    img_path = os.path.join(path, image)
    print(img_path)
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    data = [image, H, W, H*W]
    img_dim_data.append(data)
    height.append(H)
    width.append(W)

print('Saving into dimensions.csv file...\n')
data_df = pd.DataFrame(img_dim_data, columns = ['img_name', 'height', 'width', 'area'])
data_df.to_csv('dimensions.csv', mode='w+', index=False)

print('Show scatter plot ...\n')
plt.scatter(height, width)
plt.xlabel('height')
plt.ylabel('width')
plt.title('Scatter plot of width against height')
plt.show()

print('DONE!\n')
