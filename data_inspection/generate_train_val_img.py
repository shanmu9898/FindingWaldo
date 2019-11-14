import os
import cv2

count_pics = 1


def extract_template(base_write_path, anno_path, read_path, class_, pic_id):
    global count_pics
    f = open(anno_path, 'r')
    xml_content = f.read().split('\n')
    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0
    counter = 0
    name_flag = False
    for line in xml_content:
        line = line.strip()
        if name_flag:
            if line[:6] == "<xmin>":
                line = line[6:]
                line = line[:-7]
                x_min = int(line)
            if line[:6] == "<xmax>":
                line = line[6:]
                line = line[:-7]
                x_max = int(line)
            if line[:6] == "<ymin>":
                line = line[6:]
                line = line[:-7]
                y_min = int(line)
            if line[:6] == "<ymax>":
                line = line[6:]
                line = line[:-7]
                y_max = int(line)
                name_flag = False
                # print("x_min:", x_min, "x_max:", x_max, "y_min:", y_min, "y_max:", y_max)

                # extract the template and save it
                img = cv2.imread(read_path)
                img = img[y_min:y_max, x_min:x_max]
                write_path = os.path.join(base_write_path, pic_id + "_" + str(counter) + ".jpg")
                cv2.imwrite(write_path, img)
                counter += 1
                print(class_, "found:", count_pics)
                count_pics += 1

        if line[:6] == '<name>':
            line = line[6:]
            line = line[:-7]
            if line == class_:
                name_flag = True
    f.close()


classes = ['waldo', 'wenda', 'wizard']
sets = ['train', 'val']
base_JPEG_path = os.path.join("..", "datasets", "JPEGImages")
base_anno_path = os.path.join("..", "datasets", "Annotations")
base_write_path = os.path.join("..", "datasets", "extracted_images")
image_set_train_path = os.path.join("..", "datasets", "ImageSets", "train.txt")
image_set_val_path = os.path.join("..", "datasets", "ImageSets", "val.txt")
train_image_list = open(image_set_train_path, 'r').read().split('\n')
val_image_list = open(image_set_val_path, 'r').read().split('\n')


if os.path.isdir(base_write_path):
    print("dir alr exists. please delete extracted_images dir before re-generating images.")
else:
    print("creating directories...")
    os.mkdir(base_write_path)  # make the directory for each class
    for class_ in classes:
        print("generating dir for", class_)
        write_path = os.path.join(base_write_path, class_)
        os.mkdir(write_path)
        list_of_images = os.listdir(base_JPEG_path)

        for set_ in sets:
            os.mkdir(os.path.join(write_path, set_))
            for images in list_of_images:
                pic_id = images[:-4]
                if (set_ == 'train' and pic_id in train_image_list) or (set_ == 'val' and pic_id in val_image_list):
                    print("currently at", set_, class_, "directory:", images)
                    image_path = os.path.join(base_JPEG_path, pic_id + '.jpg')
                    anno_path = os.path.join(base_anno_path, pic_id + '.xml')
                    extract_template(os.path.join(write_path, set_), anno_path, image_path, class_, pic_id)
            count_pics = 1
