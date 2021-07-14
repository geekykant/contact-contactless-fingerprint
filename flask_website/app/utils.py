from os.path import join, dirname, basename, exists
from os import makedirs
from glob import glob
import cv2
import numpy as np

dirname = dirname(__file__)
db_path = join(dirname, 'static/db_fingerprints')

def getAllImagesFromDatabase():
    fingerprint_db = np.load(join(dirname, 'real_images.npy'))

    db_json = []
    for file in fingerprint_db:
        id = int(basename(file).split('_')[0])
        name = "Person_%d" %(id)
        file =  file.replace('Users/sreekant/Documents/dev/fingerdash/app/contact_dataset', 'dataset')
        db_json.append({"id": id, "label": name, "url": file})

    return sorted(db_json, key=lambda k: k['id'])
