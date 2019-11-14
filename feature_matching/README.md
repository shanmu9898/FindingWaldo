# Feature Matching

Explanation and sample results have been provided in the report.

This README.md file is to show how to run each of the file in this directory.

**show_sift_keypoints.py**

```shell
$python show_sift_keypoints.py -p {path to image} -n {number of highest response keypoints to show}
```

**show_sift_match.py**

```shell
$python show_sift_match.py -p1 {path to first_image} -p2 {path to second_image}
```

**feature_matching.py**

Be careful when you run the following command. Potentially a lot of images will be generated in this directory. You may want to kill the process before its completion using `Ctrl + C`.

```shell
$python feature_matching.py -t {path to template_image} -p {path to test_image}
```

Note that you may need to create a conda environment using the `../requirements.txt` to run the three files in this directory.
