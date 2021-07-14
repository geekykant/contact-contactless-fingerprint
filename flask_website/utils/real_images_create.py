from os.path import join, basename, dirname
from glob import glob
import numpy as np

dirname = dirname(__file__)
contact_dataset_path = '/Users/sreekant/Documents/dev/fingerdash/app/contact_dataset'

def make_real_images_npy():
    import random
    npy_images = np.array([])
    # all_images = glob(join(contact_dataset_path, "**/*.jpg"))

    for  person_id in range(1, 336 + 1):
        imgs = glob(join(contact_dataset_path, f"**/{person_id}_*.jpg"))
        one_image =  random.choice(imgs)
        npy_images = np.append(npy_images, one_image)

    np.save(join(dirname, "../app/real_images.npy"), npy_images)

if __name__ == '__main__':
    make_real_images_npy()
