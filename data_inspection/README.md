# Data Inspection

Explanation and sample results have been provided in the report.

This README.md file is to show how to run each of the file in this directory.

**plot_height_width.py**

```shell
$python -p {path to directory with all the images (no sub-directory)}
```

**generate_train_val_img.py**

Make sure that `../datesets/extracted_images/` directory is not present before running the following command.

```shell
$python generate_train_val_img.py
```

This will create `../datesets/extracted_images/` directory.


**visualize_hog_img.py**

This is not mentioned in the report. The purpose of this script is to visualise hog features of the training images. This script will write into the `hog` sub-directory in this directory.

```shell
$python visualize_hog_img.py -p {path to directories of images}
```
