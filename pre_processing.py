import cv2
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
import random
from skimage import util, filters, morphology
from os.path import basename, join
import numpy as np


# global variables
EXPAND_WIDTH = 250
EXPAND_HEIGHT = 250

CropWidth = 220
CropHeight = 220
ExpWidth = 250
ExpHeight = 250

# get fingerprint region for crop
def get_fp_region(img_path, crop_width=250, crop_height=250):
    CropWidth = crop_width
    CropHeight = crop_height
    ExpWidth = EXPAND_WIDTH
    ExpHeight = EXPAND_HEIGHT

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    thresh = filters.threshold_otsu(image)

    picBW = image > thresh
    # plot_image(picBW, "B&W")

    bw = morphology.closing(image > thresh, morphology.square(3))
    # plot_image(bw, "B&W")

    cleared = bw.copy()

    img_width = image.shape[1]
    img_height = image.shape[0]
    #print(img_width, img_height)

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

def makeTrainImages(img_path, finger_id, j, save_url = ''):
    finger_idx = 1
    # get fingerprint region
    img = Image.open(img_path)
    (crop_l, crop_t, crop_r, crop_b) = get_fp_region(img_path)

    # crop for process image
    crop_x = (ExpWidth - CropWidth) / 2
    crop_y = (ExpHeight - CropHeight) / 2
    img = img.crop([crop_l, crop_t, crop_r, crop_b])

    # plot_image(np.array(img), "ROI Fingerprint")

    # single crop - save
    img_c = img.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
    img_path_new = join(save_url, f'{finger_id}_{j}' + '{0:03d}'.format(finger_idx) + '.jpg')

    # plot_image(np.array(img_c), "ROI Fingerprint")

    # img_arr = np.array(img_c)
    # out = ~fingerprint_enhancer.enhance_Fingerprint(img_arr)

    # img_c = Image.fromarray(out)
    img_c.save(img_path_new)

    finger_idx += 1

    # rotate crop
    for i in range(1):
        ang = random.randint(10, 350)
        img_rot = img.rotate(ang)
        img_c = img_rot.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
        img_arr = np.array(img_c)
        img_path_new = join(save_url, f'{finger_id}_{j}' + '{0:03d}'.format(finger_idx) + '.jpg')
        img_c.save(img_path_new)
        finger_idx += 1

    def change_contrast(img):
      image = np.asarray(img)
      adjusted = cv2.equalizeHist(image)
      return Image.fromarray(adjusted)

    # auto contrast & crop
    for i in range(3):
        ang = random.randint(10, 350)
        img_autocont = change_contrast(img)
        img_rot = img_autocont.rotate(ang)
        img_c = img_rot.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
        img_arr = np.array(img_c)
        img_path_new = join(save_url, f'{finger_id}_{j}' + '{0:03d}'.format(finger_idx) + '.jpg')
        img_c.save(img_path_new)
        finger_idx += 1

    # noise & contact and crop
    for i in range(3):
        ang = random.randint(10, 350)
        img_autocont = change_contrast(img)
        img_rot = img_autocont.rotate(ang)
        img_arr = np.array(img_rot)
        util.random_noise(img_arr, mode = 'gaussian')
        img_rot = Image.fromarray(img_arr)
        img_c = img_rot.crop([crop_x, crop_y, crop_x + CropWidth, crop_y + CropHeight])
        img_arr = np.array(img_c)
        img_path_new = join(save_url, f'{finger_id}_' + '{0:03d}'.format(finger_idx) + '.jpg')
        img_c.save(img_path_new)
        finger_idx += 1

from glob import glob
contact_images = '/content/drive/MyDrive/#Fingerprint DeepLearning/Cross_Fingerprint_Images_Database/contact-based_fingerprints'
from os.path import join

for finger_id in range(1, 336 + 1):
    print(f"[*] creating {finger_id}...")
    all_images = glob(join(contact_images, '**/*.jpg'))
    
    for j, img_path in enumerate(all_images):
        makeTrainImages(img_path, finger_id, j + 1, '')
