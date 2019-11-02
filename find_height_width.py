import os
import cv2
import matplotlib.pyplot as plt
import argparse

def get_image_paths(path):
    image_paths = []
    image_names = os.listdir(path)
    
    for image_name in image_names:
        if image_name == '.DS_Store':
            continue
        img_path = os.path.join(path, image_name)
        image_paths.append(img_path)

    return image_paths


def get_image_height_width(path):
    img = cv2.imread(path)
    h, w, _ = img.shape
    return h, w


def get_boundary_values_counts(values, bin_size):
    max_val = max(values)
    min_val = min(values)
    print(max_val, min_val)
    labels = []
    boundaries = []

    val = round(min_val) # lower bound
    while val < max_val:
        label = f'{val}-{val+bin_size}'
        labels.append(label)
        val += bin_size
        boundaries.append(val)

    counts = [0] * len(boundaries)
    for value in values:
        for index, boundary in enumerate(boundaries):
            if value <= boundary:
                counts[index] += 1
                break

    return labels, counts


def plot_bar(x_axis, y_axis, title):
    plt.clf()
    plt.bar(x_axis, y_axis)
    plt.xlabel('Range', fontsize=5)
    plt.ylabel('Count', fontsize=5)
    plt.title(title)
    plt.savefig(f'{title}.png')
    plt.show()


def print_vals(labels, vals):
    for label, val in zip(labels, vals):
        print(f'{label}: {val}')

def plot_distribution(path, bin_size):
    '''
    The path must be a directory with the following structure:
    directory
    ├── waldo
    ├── wenda
    └── wizard
    '''
    categories = ['waldo', 'wenda', 'wizard']
    heights = []
    widths = []
    for category in categories:
        category_path = os.path.join(path, category)
        image_paths = get_image_paths(category_path)

        for image_path in image_paths:
            h, w = get_image_height_width(image_path)
            heights.append(h)
            widths.append(w)

    height_labels, height_counts = get_boundary_values_counts(heights, bin_size)
    width_labels, width_counts = get_boundary_values_counts(widths, bin_size)

    print_vals(height_labels, height_counts)
    print_vals(height_labels, height_counts)

    plot_bar(height_labels, height_counts, 'height')
    plot_bar(width_labels, width_counts, 'width')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot graph of height and weight')
    parser.add_argument('-p', action='store', 
                        type=str, required=True, help='Path to images')
    parser.add_argument('-b', action='store',
                        type=int, required=True, help='Bin size')

    args = vars(parser.parse_args())
    path = args['p']
    bin_size = args['b']

    plot_distribution(path, bin_size)
