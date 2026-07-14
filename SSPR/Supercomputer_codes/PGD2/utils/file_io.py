from tifffile import imread, imwrite
import glob
import cupy as cp
from natsort import natsorted


def load_img_data(img_file):
    """Loads an image"""
    img_data = cp.array(imread(img_file), dtype=cp.float32)
    return img_data


def write_img_data(img_file, img_data):
    """Saves an image"""
    imwrite(img_file, img_data, photometric='minisblack')


def load_multi_img_data(img_files):
    """Loads multiple images"""
    print("Reading ", len(img_files), " images")
    img_data = glob.glob(img_files)
    img_data = natsorted(img_data)
    img_data = cp.array(img_data, dtype=cp.float32)
    return img_data
