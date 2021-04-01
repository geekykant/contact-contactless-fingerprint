from os.path import join, dirname, basename
from glob import glob
import cv2
import numpy as np

dirname = dirname(__file__)
db_path = join(dirname, 'static/db_fingerprints')

def getAllImagesFromDatabase():
    fingerprint_db = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        fingerprint_db.extend(glob(join(db_path, ext)))

    db_json = []
    id = 1
    for file in fingerprint_db:
        name = basename(file).rsplit('.', 1)[0]
        db_json.append({"id": id, "label": name, "url": file.replace(dirname, '')})
        id += 1

    return sorted(db_json, key=lambda k: k['label'])

def saveImageToDatabase(label, img):
    # img = cropCircleImage(img)
    cv2.imwrite(join(db_path, label + ".jpg"), img)

# def cropCircleImage(img, crop_size=40):
#     h,w = img.shape
#     mask = np.zeros((h,w), np.uint8)
#
#     cs = crop_size
#     x = (w + h)//2
#     r = h//2 - cs
#
#     cv2.circle(mask, (h//2, w//2), r, (255,255,255), -1)
#     img = cv2.bitwise_and(img, img, mask=mask)
#     cropped_image = img[cs:-cs, cs:-cs]
#     return cropped_image
