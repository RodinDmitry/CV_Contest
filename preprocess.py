import numpy as np
import os
import cv2
from tqdm import tqdm


def process_file(file_name, result_name):
    image = cv2.imread(file_name)
    h, w, c = image.shape
    crop_size = 224
    if h > w:
        new_w = 255
        new_h = max(int(255 * h / w), crop_size)
    else:
        new_h = 255
        new_w = max(int(255 * w / h), crop_size)
    image = cv2.resize(image, (new_w, new_h))

    crop_offset_w = int((new_w - crop_size) / 2)
    crop_offset_h = int((new_h - crop_size) / 2)
    image = image[crop_offset_h:crop_offset_h + crop_size, crop_offset_w:crop_offset_w + crop_size, :]
    assert image.shape[0] == crop_size
    assert image.shape[1] == crop_size
    cv2.imwrite(result_name, image)


def process_folder(source_folder, target_folder):
    files = os.listdir(source_folder)
    for file in tqdm(files):
        file_name = os.path.join(source_folder, file)
        result_name = os.path.join(target_folder, file)
        process_file(file_name, result_name)


def process_dataset(source_folder, result_folder):
    files = os.listdir(source_folder)
    for file in tqdm(files):
        subfolder_name = os.path.join(source_folder, file)
        result_subfolder_name = os.path.join(result_folder, file)
        try:
            os.mkdir(result_subfolder_name)
        except Exception as e:
            print(e)
        process_folder(subfolder_name, result_subfolder_name)


if __name__ == '__main__':
    file_name = r'D:\Datasets\test_data'
    result_name = r'D:\Datasets\test_data_processed'
    process_dataset(file_name, result_name)
