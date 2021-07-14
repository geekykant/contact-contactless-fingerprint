import cv2
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
import random
from skimage import util, filters, morphology
from os.path import basename, join
import numpy as np

# global variables
EXPAND_WIDTH = 220
EXPAND_HEIGHT = 220

CropWidth = 220
CropHeight = 220
ExpWidth = 220
ExpHeight = 220

# get fingerprint region for crop
def get_fp_region(image, crop_width=220, crop_height=220):
    CropWidth = crop_width
    CropHeight = crop_height
    ExpWidth = EXPAND_WIDTH
    ExpHeight = EXPAND_HEIGHT

    thresh = filters.threshold_otsu(image)

    picBW = image > thresh
    bw = morphology.closing(image > thresh, morphology.square(3))

    cleared = bw.copy()

    img_width = image.shape[1]
    img_height = image.shape[0]

    crop_l = img_width
    crop_r = 0
    crop_t = img_height
    crop_b = 0
    for i in range(img_height):
        for j in range(img_width):
            if cleared[i, j] == False:
                if (crop_l > j):
                    crop_l = j
                if (crop_r < j):
                    crop_r = j
                if (crop_t > i):
                    crop_t = i
                if (crop_b < i):
                    crop_b = i

    if ((crop_r - crop_l) < CropWidth):
        diff = CropWidth - (crop_r - crop_l)
        if (crop_r + crop_l > CropWidth): # right
            if (img_width - crop_r > diff / 2):
                crop_r += diff / 2
                crop_l -= diff / 2
            else:
                crop_r = img_width - 1
                crop_l = crop_r - (CropWidth + 2)
        else: # left
            if (crop_l > diff / 2):
                crop_l -= diff / 2
                crop_r += diff / 2
            else:
                crop_l = 1
                crop_r = crop_l + (CropWidth + 2)
    if ((crop_b - crop_t) < CropHeight):
        diff = CropHeight - (crop_b - crop_t)
        if (crop_b + crop_t > CropHeight): # bottom
            if (img_height - crop_b > diff / 2):
                crop_b += diff / 2
                crop_t -= diff / 2
            else:
                crop_b = img_height - 1
                crop_t = crop_b - (CropHeight + 2)
        else: # top
            if (crop_t > diff / 2):
                crop_t -= diff / 2
                crop_b += diff / 2
            else:
                crop_t = 1
                crop_b = crop_t + (CropHeight + 2)

    # expand region for rotation
    crop_l = (crop_r + crop_l - CropWidth) / 2
    crop_r = crop_l + CropWidth
    crop_t = (crop_t + crop_b - CropHeight) / 2
    crop_b = crop_t + CropHeight
    crop_l = (int)(crop_l - ((ExpWidth - CropWidth) / 2))
    crop_r = (int)(crop_r + ((ExpWidth - CropWidth) / 2))
    crop_t = (int)(crop_t - ((ExpHeight - CropHeight) / 2))
    crop_b = (int)(crop_b + ((ExpHeight - CropHeight) / 2))

    # check expanded region
    diff = 0
    if (crop_l < 0):
        diff = 0 - crop_l
        crop_l = crop_l + diff
        crop_r = crop_r + diff
    if (crop_r >= img_width):
        diff = crop_r - (img_width - 1)
        crop_l = crop_l - diff
        crop_r = crop_r - diff

    diff = 0
    if (crop_t < 0):
        diff = 0 - crop_t
        crop_t = crop_t + diff
        crop_b = crop_b + diff
    if (crop_b >= img_height):
        diff = crop_b - (img_height - 1)
        crop_t = crop_t - diff
        crop_b = crop_b - diff

    return (crop_l, crop_t, crop_r, crop_b)

def cropToRegionInterest(image_arr):
    (crop_l, crop_t, crop_r, crop_b) = get_fp_region(image_arr)

    # crop for process image
    crop_x = (ExpWidth - CropWidth) / 2
    crop_y = (ExpHeight - CropHeight) / 2
    img = Image.fromarray(image_arr, 'L')
    img = img.crop([crop_l, crop_t, crop_r, crop_b])

    # single crop - save
    img_c = img.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
    return np.array(img_c)

import cv2
from albumentations import (
    Compose, CenterCrop, ShiftScaleRotate
)

# AUGMENTATIONS_TRAIN = Compose([
#     CenterCrop (220, 220, always_apply=True, p=1.0),
#     ShiftScaleRotate(rotate_limit=(-20,20), scale_limit=0.15,
#         interpolation=0, border_mode=0, value=(255, 255, 255), mask_value=None, p=0.7),
# ])

AUGMENTATIONS_TEST = Compose([
    CenterCrop (220, 220, always_apply=True, p=1.0),
])

def enhanceAgain(img):
    return (AUGMENTATIONS_TEST(image = img)['image']*255).astype(np.uint8)
