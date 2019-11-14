from pylab import imshow, show
import numpy as np
import cv2
import copy


def filter_for_white(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    w00 = (70, 50, 255)
    w01 = (0, 0, 200)
    mask0 = cv2.inRange(hsv_img, w01, w00)

    w00 = (175, 90, 255)
    w01 = (150, 30, 200)
    mask1 = cv2.inRange(hsv_img, w01, w00)

    w00 = (12, 120, 255)
    w01 = (0, 40, 240)
    mask2 = cv2.inRange(hsv_img, w01, w00)

    mask_final = mask0 + mask1 + mask2

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_final)

    one_channel = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    white_area = copy.deepcopy(one_channel)

    white_threshold = one_channel == 0
    white_area[white_threshold] = 0

    white_threshold = one_channel > 0
    white_area[white_threshold] = 1

    return white_area


def filter_for_red(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lower_red0 = (0, 130, 190)
    lower_red1 = (5, 255, 255)
    mask = cv2.inRange(hsv_img, lower_red0, lower_red1)

    upper_red10 = (170, 130, 190)
    upper_red11 = (180, 150, 255)
    mask3 = cv2.inRange(hsv_img, upper_red10, upper_red11)

    upper_red0 = (170, 170, 190)
    upper_red1 = (180, 255, 255)
    mask2 = cv2.inRange(hsv_img, upper_red0, upper_red1)

    # mask2 extended
    upper_red3 = (170, 170, 140)
    upper_red31 = (180, 255, 190)
    mask4 = cv2.inRange(hsv_img, upper_red3, upper_red31)

    mask_super = mask + mask2 + mask3 + mask4

    result = cv2.bitwise_and(rgb_img, rgb_img, mask=mask_super)

    one_channel = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    red_area = copy.deepcopy(one_channel)

    red_threshold = one_channel == 0
    red_area[red_threshold] = 0

    red_threshold = one_channel > 0
    red_area[red_threshold] = 1

    return red_area


def find_shirt(rgb, red_img, white_img):
    translation_matrix = np.float32([[1, 0, 0], [0, 1, 7]])
    shifted_red_down = cv2.warpAffine(red_img, translation_matrix, (red_img.shape[1], red_img.shape[0]))

    translation_matrix = np.float32([[1, 0, 0], [0, 1, -7]])
    shifted_red_up = cv2.warpAffine(red_img, translation_matrix, (red_img.shape[1], red_img.shape[0]))


    result = cv2.bitwise_and(white_img, white_img, mask=shifted_red_up)
    result2 = cv2.bitwise_and(white_img, white_img, mask=shifted_red_down)

    up_sum = np.sum(np.reshape(result, (-1)))
    down_sum = np.sum(np.reshape(result2, (-1)))

    sum_up_down = up_sum + down_sum

    min_sum = 10
    ############################################################################################################################
    # shift left and right
    translation_matrix = np.float32([[1, 0, -5], [0, 1, 0]])
    shifted_red_left = cv2.warpAffine(red_img, translation_matrix, (red_img.shape[1], red_img.shape[0]))

    translation_matrix = np.float32([[1, 0, 5], [0, 1, 0]])
    shifted_red_right = cv2.warpAffine(red_img, translation_matrix, (red_img.shape[1], red_img.shape[0]))

    result = cv2.bitwise_and(white_img, white_img, mask=shifted_red_left)
    result2 = cv2.bitwise_and(white_img, white_img, mask=shifted_red_right)

    overall_result = result + result2

    sum_left_right = np.sum(np.reshape(overall_result, (-1)))
    ############################################################################################################################


    if min(up_sum, down_sum) > max(up_sum, down_sum) / 3 and up_sum > min_sum and down_sum > min_sum\
            and sum_left_right < sum_up_down / 2:
        return True


# takes in img in BGR color space
# returns list of coordinates and images of possible waldo locations.
# note that list of images returned are also in BGR
def detect_stripes(img):

    og = img
    rgb_img = cv2.cvtColor(og, cv2.COLOR_BGR2RGB)

    toErase = copy.deepcopy(rgb_img)

    print("getting red image...")
    red_img = filter_for_red(og)
    print("obtained red. getting white image...")
    white_img = filter_for_white(og)
    print("obtained white.")

    height, width = red_img.shape
    print("height", height, "width", width)

    # dimension of window

    window_h = 50
    window_w = 50

    # number of small boxes across x and y to obtain
    num_x = width // window_w
    num_y = height // window_h

    coordinates = []

    for i in range(num_y):

        start_y = i * window_h
        end_y = start_y + window_h

        for j in range(num_x):

            start_x = j * window_w
            end_x = start_x + window_w

            red_cropped_image = red_img[start_y:end_y, start_x:end_x]
            white_cropped_image = white_img[start_y:end_y, start_x:end_x]
            rgb_cropped_image = rgb_img[start_y:end_y, start_x:end_x]

            isHotSpot = find_shirt(rgb_cropped_image, red_cropped_image, white_cropped_image)

            if isHotSpot:
                coordinates.append([start_y, start_x])
            else:
                toErase[start_y:end_y, start_x:end_x] *= 0

    print("total number of segments:", num_y * num_x)
    print("number of segments remaining:", len(coordinates))

    imshow(toErase)
    show()

    sw_h = 250
    sw_w = 250

    image_list = []
    coordinate_list = []

    for i in range(0, height-sw_h, sw_h):
        for j in range(0, width-sw_w, sw_w):

            waldo_detected = False
            for k in range(len(coordinates)):
                if coordinates[k][0] >= i and coordinates[k][0] <= i + sw_h\
                        and coordinates[k][1] >= j and coordinates[k][1] <= j + sw_w:
                    waldo_detected = True
                    break

            if waldo_detected:
                image_list.append(og[i:i+sw_h, j:j+sw_w])
                coordinate_list.append([i,j])

    return image_list, coordinate_list
