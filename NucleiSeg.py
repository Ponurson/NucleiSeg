import os
from joblib import Parallel, delayed
from skimage import io
import numpy as np
import cv2
from skimage import feature
import scipy


def imfill(closing):
    im_floodfill = closing.copy()
    h, w = closing.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = closing | im_floodfill_inv
    return im_out


def max_rozmiar(tmp, tmax):
    max_x_size = 0
    max_y_size = 0
    for t in range(tmax):
        labeled_object = cv2.connectedComponentsWithStats(tmp[t], 4, cv2.CV_32S)
        if labeled_object[0] > 1:
            labeled_object_info = labeled_object[2]
            max_x_size = np.max([max_x_size, labeled_object_info[1, 2]])
            max_y_size = np.max([max_y_size, labeled_object_info[1, 3]])
    return max_x_size, max_y_size


def im_roll_for_cropping(im_2d, im_4d):
    labeled_object = cv2.connectedComponentsWithStats(im_2d, 4, cv2.CV_32S)
    labeled_object_info = labeled_object[2]
    center_y = int(labeled_object_info[1, 0] + np.floor(labeled_object_info[1, 2] / 2))
    center_x = int(labeled_object_info[1, 1] + np.floor(labeled_object_info[1, 3] / 2))
    im_2d = np.roll(im_2d, ((256 - center_x), (256 - center_y)), axis=(0, 1))
    im_4d = np.roll(im_4d, ((256 - center_x), (256 - center_y)), axis=(2, 3))
    return im_2d, im_4d


def load_images_from_number_brightness_method(path):
    list_of_filenames = []
    for file in os.listdir(path + "/"):
        if file.endswith(".tif"):
            list_of_filenames.append(file)
    list_of_images = []
    for i in list_of_filenames:
        list_of_images.append(io.imread(path + "/" + i))
    return list_of_images, list_of_filenames


def segmentation_through_edge_detection(mean_projection_in_second_channel):
    edge_images_partially_filled_list = []
    for i in range(15, 20):
        edge_im = feature.canny(mean_projection_in_second_channel, sigma=i)
        edge_images_partially_filled_list.append(imfill(np.uint8(edge_im * 255)))

    edge_im_partially_filled = sum(edge_images_partially_filled_list)
    kernel = np.ones((3, 3), np.uint8)
    im_with_removed_nonfilled_objects = cv2.morphologyEx(imfill(edge_im_partially_filled), cv2.MORPH_OPEN, kernel)
    return im_with_removed_nonfilled_objects


def nuc_seg_from_live_data_spinning_disc(path):
    image_stack_with_timeframes = io.imread(path)
    nuclei_cropped_4d_stacks = []

    dimensions = image_stack_with_timeframes.shape
    frames_num, im_in_stack_num, num_of_channels = dimensions[0], dimensions[1], dimensions[2]
    timeframes_mean_projection_in_second_channel = np.mean(image_stack_with_timeframes[:, :, num_of_channels - 1, :, :],
                                                           axis=1)
    mask_for_timeframes = np.zeros([dimensions[0], dimensions[3], dimensions[4]])
    for t in range(frames_num):

        im_with_removed_nonfilled_objects = segmentation_through_edge_detection(
            timeframes_mean_projection_in_second_channel[t])
        labeled_objects = cv2.connectedComponentsWithStats(im_with_removed_nonfilled_objects, 4, cv2.CV_32S)
        labeled_objects_image = labeled_objects[1]
        labeled_objects_info = labeled_objects[2]
        for i in range(1, labeled_objects_info.shape[0]):
            if labeled_objects_info[i][4] > 7000 or labeled_objects_info[i][4] < 1000:
                labeled_objects_image[labeled_objects_image == i] = 0
        mask_for_timeframes[t] = labeled_objects_image

    labeled_nuclei_time_continous_data = scipy.ndimage.label(mask_for_timeframes)
    labeled_nuclei_time_continous_images = labeled_nuclei_time_continous_data[0]
    for i in range(labeled_nuclei_time_continous_images.max()):
        single_nuclei_with_time = np.uint8(labeled_nuclei_time_continous_images.copy())
        single_nuclei_with_time[single_nuclei_with_time != i] = 0
        max_x_size, max_y_size = max_rozmiar(single_nuclei_with_time, frames_num)
        single_nuclei_cropped = []
        for t in range(frames_num):
            if single_nuclei_with_time[t].max() > 0:
                tmpt, imgt = im_roll_for_cropping(labeled_objects_image[t], image_stack_with_timeframes[t])
                labeled_objects = cv2.connectedComponentsWithStats(tmpt, 4, cv2.CV_32S)
                labeled_objects_info = labeled_objects[2]
                x_start, y_start = labeled_objects_info[1, 0] - 10, labeled_objects_info[1, 1] - 10
                x_end = x_start + max_x_size + 20
                y_end = y_start + max_y_size + 20
                single_nuclei_cropped.append(imgt[:, :, y_start:y_end, x_start:x_end])
        if len(single_nuclei_cropped) >= 10:
            nuclei_cropped_4d_stacks.append(np.asarray(single_nuclei_cropped))

    return nuclei_cropped_4d_stacks


def nuc_seg_from_immuno_spinning_disc(path):
    stack_of_images = io.imread(path)

    if stack_of_images.ndim != 3:
        mean_projection_in_second_channel = np.mean(stack_of_images[1], axis=0)
    else:
        mean_projection_in_second_channel = stack_of_images[:, :, 1]

    im_with_removed_nonfilled_objects = segmentation_through_edge_detection(mean_projection_in_second_channel)
    labeled_objects = cv2.connectedComponentsWithStats(im_with_removed_nonfilled_objects, 4, cv2.CV_32S)
    labeled_objects_info = labeled_objects[2]
    nuclei_cropped_3d_stacks = []
    for i in range(0, labeled_objects_info.shape[0]):
        if labeled_objects_info[i][4] < 7000 and labeled_objects_info[i][4] > 3000:
            x_start = labeled_objects_info[i][0] - 20
            y_start = labeled_objects_info[i][1] - 20
            x_end = x_start + labeled_objects_info[i][2] + 40
            y_end = y_start + labeled_objects_info[i][3] + 40
            nuclei_cropped_3d_stacks.append(stack_of_images[:, :, y_start:y_end, x_start:x_end])
    return nuclei_cropped_3d_stacks, labeled_objects


def load_nuc_seg_images(path):
    # all names
    list_of_filenames = []
    for file in os.listdir(path + "/"):
        if file.endswith(".tif") or file.endswith(".jpg"):
            list_of_filenames.append(file)
    list_of_images = []

    for i in list_of_filenames:
        list_of_images.append(io.imread(path + "/" + i))  # list of images readed as arrays
    return list_of_images, list_of_filenames


def segmentation_controller(mode, path):
    if (mode == 'tile'):
        list_of_images, labeled_map = nuc_seg_from_immuno_spinning_disc(path)
        return list_of_images, labeled_map
    elif (mode == 'live'):
        list_of_pathnames = []
        list_of_filenames = []
        for file in os.listdir(path + "/"):
            if file.endswith(".tif"):
                list_of_pathnames.append(path + "/" + file)
                list_of_filenames.append(file)

        helper_list_of_images = Parallel(n_jobs=-1, max_nbytes=None)(
            delayed(nuc_seg_from_live_data_spinning_disc)(a) for a in list_of_pathnames)

        list_of_images = []
        for i in range(0, len(helper_list_of_images)):
            list_of_images = list_of_images + helper_list_of_images[i]

        return list_of_images, list_of_filenames

    elif (mode == 'data'):
        list_of_images, list_of_filenames = load_nuc_seg_images(path)
    elif (mode == 'NB'):
        list_of_images, list_of_filenames = load_images_from_number_brightness_method(path)

    return list_of_images, list_of_filenames
