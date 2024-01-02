from PIL import Image, ImageFilter
import numpy as np
import os

def get_means(array_2d):
    row_means = np.mean(array_2d, axis=1)
    column_means = np.mean(array_2d, axis=0)
    return row_means.astype(np.int8), column_means.astype(np.int8)

def get_stds(array_2d):
    row_std_devs = np.std(array_2d, axis=1)
    column_std_devs = np.std(array_2d, axis=0)
    return row_std_devs.astype(np.int8), column_std_devs.astype(np.int8)

def preprocess(img):
    img_hsv = img.convert("HSV")
    saturation_channel = np.array(img_hsv)[:, :, 1]
    img_gray = np.array(img.convert("L"))
    new_image = np.pad(img_gray, (0,6), mode='constant')
    contour_img = np.array(img.convert("L").filter(ImageFilter.CONTOUR))

    mean_saturation_row, mean_saturation_col = get_means(saturation_channel)
    std_saturation_row, std_saturation_col = get_stds(saturation_channel)
    new_image[0:200, 200] = mean_saturation_row
    new_image[0:200, 201] = std_saturation_row
    new_image[200, 0:200] = mean_saturation_col
    new_image[201, 0:200] = std_saturation_col

    mean_gray_row, mean_gray_col = get_means(img_gray)
    std_gray_row, std_gray_col = get_stds(img_gray)
    new_image[0:200, 202] = mean_gray_row
    new_image[0:200, 203] = std_gray_row
    new_image[202, 0:200] = mean_gray_col
    new_image[203, 0:200] = std_gray_col

    contour_img = contour_img / 255
    contour_img = contour_img * contour_img
    contour_img = contour_img * contour_img
    contour_img = contour_img * contour_img
    contour_img = contour_img * 255

    mean_contour_row, mean_contour_col = get_means(contour_img)
    std_contour_row, std_contour_col = get_stds(contour_img)
    new_image[0:200, 204] = mean_contour_row
    new_image[0:200, 205] = std_contour_row
    new_image[204, 0:200] = mean_contour_col
    new_image[205, 0:200] = std_contour_col

    out = Image.fromarray(new_image, 'L')

    return out