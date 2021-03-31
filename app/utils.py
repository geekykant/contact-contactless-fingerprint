from os.path import join, dirname, basename
from glob import glob

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
    import cv2
    cv2.imwrite(join(db_path, label + ".jpg"), img)
