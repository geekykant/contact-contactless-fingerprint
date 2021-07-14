import os, sys
from . import predictor
from . import enhancer
from . import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import cv2, jsonpickle, numpy as np

# start flask
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 #2 MB

@app.route('/dataset/<path:filepath>')
def download_file(filepath):
    filename = os.path.basename(filepath)
    MEDIA_FOLDER = '/Users/sreekant/Documents/dev/fingerdash/app/contact_dataset/first_session'
    if "first_session" not in filepath:
        MEDIA_FOLDER = '/Users/sreekant/Documents/dev/fingerdash/app/contact_dataset/second_session'
    return send_from_directory(MEDIA_FOLDER, filename)

# render default webpage
@app.route('/')
def home():
    stored_fp = utils.getAllImagesFromDatabase()
    return render_template('database.html', fps=stored_fp, detection_page=False, title="Fingerprint Prediction")

# render default analyze page
@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/predict_with_db', methods=['POST'])
def predictWithDb():
    if request.method == 'POST':
        fp = request.files['fp1'].read()

        fpimg = np.frombuffer(fp, np.uint8)
        fpimg = cv2.imdecode(fpimg, cv2.IMREAD_GRAYSCALE)

        fpimg = enhancer.cropToRegionInterest(fpimg)
        fpimg = enhancer.enhanceAgain(fpimg)

        # cv2.imwrite('out.jpg', out_img)
        pred_result = None

        try:
            pred_result = predictor.getPredictionDb(fpimg)
            if pred_result['best_pred_pred'] is not None:
                pred_result['best_pred_pred'] = float("%.3f" %(pred_result['best_pred_pred']))
        except Exception as e:
            print(e)
            return Response(response={'status': 'Predction Problem! Check image size, maybe.'}, status=503, mimetype="application/json")

        response_pickled = jsonpickle.encode(pred_result)
        return Response(response=response_pickled, status=200, mimetype="application/json")

# render database webpage displaying all stored fingerprints
@app.route('/database')
def database_home():
    stored_fp = utils.getAllImagesFromDatabase()
    return render_template('database.html', fps=stored_fp, detection_page=True, title="Fingerprint Database")

# when the post method detect, then redirect to success function - size (356, 328)
# @app.route('/two_image_prediction', methods=['POST', 'GET'])
# def get_data():
#     if request.method == 'POST':
#         fp1 = request.files['fp1'].read()
#         fp2 = request.files['fp2'].read()
#
#         fp1img = np.frombuffer(fp1,np.uint8)
#         fp2img = np.frombuffer(fp2,np.uint8)
#
#         fp1img = cv2.imdecode(fp1img,cv2.IMREAD_GRAYSCALE)
#         fp2img = cv2.imdecode(fp2img,cv2.IMREAD_GRAYSCALE)
#
#         #resize image into 356, 328 if not.
#         if fp1img.shape != (356, 328):
#             fp1img = cv2.resize(fp1img, (356, 328))
#         if fp2img.shape != (356, 328):
#             fp2img = cv2.resize(fp2img, (356, 328))
#
#         prediction_result = -1
#
#         try:
#             prediction = predictor.two_image_prediction(fp1img, fp2img) * 100
#             prediction_result = float("%.5f" %prediction)
#         except Exception as e:
#             print(e)
#             return Response(response={'status': 'Predction Error'}, status=503, mimetype="application/json")
#
#         response = {'accuracy': prediction_result, 'status': 'up & running'}
#         response_pickled = jsonpickle.encode(response)
#         return Response(response=response_pickled, status=200, mimetype="application/json")

# when the post method detect, then redirect to success function
# @app.route('/upload_to_db', methods=['POST'])
# def store_fingerprint():
#     if request.method == 'POST':
#         default_label = 'not_labelled'
#         label = request.form.get('fp_label', default_label)
#         fp = request.files['fp1'].read()
#
#         fpimg = np.frombuffer(fp, np.uint8)
#         fpimg = cv2.imdecode(fpimg, cv2.IMREAD_GRAYSCALE)
#
#         enhanced_image = enhancer.basicEnhancing(fpimg)
#         enhanced_image = enhancer.advancedEnhancing(enhanced_image)
#         utils.saveImageToDatabase(label, enhanced_image)
#
#         response = {'status': 'Saved successfully!'}
#         response_pickled = jsonpickle.encode(response)
#         return Response(response=response_pickled, status=200, mimetype="application/json")

# @app.route('/predict_with_db', methods=['POST'])
# def predictWithDb():
#     if request.method == 'POST':
#         fp = request.files['fp1'].read()
#
#         fpimg = np.frombuffer(fp, np.uint8)
#         fpimg = cv2.imdecode(fpimg, cv2.IMREAD_GRAYSCALE)
#
#         out1 = enhancer.basicEnhancing(fpimg)
#         main_img = enhancer.advancedEnhancing(out1)
#         all_db_imgs = utils.getAllImagesFromDatabase()
#
#         pred_result = None
#         try:
#             pred_result = predictor.getPredictionDb(main_img, all_db_imgs)
#             pred_result['best_pred']['accuracy'] = float("%.5f" %(pred_result['best_pred']['accuracy']))
#         except Exception as e:
#             print(e)
#             return Response(response={'status': 'Predction Problem! Check image size, maybe.'}, status=503, mimetype="application/json")
#
#         response_pickled = jsonpickle.encode(pred_result)
#         return Response(response=response_pickled, status=200, mimetype="application/json")

#get all stored fingerprints database -> json
# @app.route('/get_db')
# def getDb():
#     response = {'status': 'successful'}
#     response['data'] = utils.getAllImagesFromDatabase()
#     response_pickled = jsonpickle.encode(response)
#     return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/test')
def test():
    import shutil
    to_copy_path = "/Users/sreekant/Documents/dev/fingerdash/app/my_dataset/wrong"
    from glob import glob
    from os.path import basename
    all_images = glob('/Users/sreekant/Documents/dev/fingerdash/app/contact_dataset/**/*.jpg')
    total_count = len(all_images)
    correct_count = 0

    all_images = list(set(all_images) - set(glob('/Users/sreekant/Documents/dev/fingerdash/app/my_dataset/*.jpg')))

    for imagePath in all_images:
        fpimg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        fpimg = enhancer.cropToRegionInterest(fpimg)
        fpimg = enhancer.enhanceAgain(fpimg)

        try:
            real_label = int(basename(imagePath).split('_')[0])
            pred_result = predictor.getPredictionDb(fpimg)
            if pred_result['best_pred_pred'] is not None and pred_result['best_pred_pred'] < 50 and pred_result['best_pred_idx'] != real_label:
                shutil.copy(imagePath, to_copy_path)
                correct_count += 1
                acc = float("%.3f" %(pred_result['best_pred_pred']))
        except Exception as e:
            print(e)

    result = f"Total count: {total_count} - Correct: {correct_count} - Wrong: {total_count-correct_count}"
    print(result)
    return Response(response=result, status=200, mimetype="text/plain")

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large (Must be less than 2MB)', 413

@app.errorhandler(503)
def request_model_not_responding(error):
    return 'Model is Not responding', 503
