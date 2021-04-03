import os, sys
from . import predictor
from . import enhancer
from . import utils

import sys
print("Python version")
print (sys.version)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2, jsonpickle, numpy as np

# start flask
app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 #16 MB

# render default webpage
@app.route('/')
def home():
    stored_fp = utils.getAllImagesFromDatabase()
    return render_template('database.html', fps=stored_fp, detection_page=False, title="Fingerprint Prediction")

# render default analyze page
@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

# when the post method detect, then redirect to success function
@app.route('/two_image_prediction', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        fp1 = request.files['fp1'].read()
        fp2 = request.files['fp2'].read()

        fp1img = np.frombuffer(fp1,np.uint8)
        fp2img = np.frombuffer(fp2,np.uint8)

        fp1img = cv2.imdecode(fp1img,cv2.IMREAD_GRAYSCALE)
        fp2img = cv2.imdecode(fp2img,cv2.IMREAD_GRAYSCALE)

        #resize image into 160 x 160 if not.
        if fp1img.shape != (160, 160):
            fp1img = cv2.resize(fp1img, (160, 160))
        if fp2img.shape != (160, 160):
            fp2img = cv2.resize(fp2img, (160, 160))

        prediction_result = -1

        try:
            prediction = predictor.two_image_prediction(fp1img, fp2img) * 100
            prediction_result = float("%.5f" %prediction)
        except Exception as e:
            print(e)
            return Response(response={'status': 'Predction Error'}, status=503, mimetype="application/json")

        response = {'accuracy': prediction_result, 'status': 'up & running'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

# render database webpage displaying all stored fingerprints
@app.route('/database')
def database_home():
    stored_fp = utils.getAllImagesFromDatabase()
    return render_template('database.html', fps=stored_fp, detection_page=True, title="Fingerprint Database")

# when the post method detect, then redirect to success function
@app.route('/upload_to_db', methods=['POST'])
def store_fingerprint():
    if request.method == 'POST':
        default_label = 'not_labelled'
        label = request.form.get('fp_label', default_label)
        fp = request.files['fp1'].read()

        fpimg = np.frombuffer(fp, np.uint8)
        fpimg = cv2.imdecode(fpimg, cv2.IMREAD_GRAYSCALE)

        enhanced_image = enhancer.basicEnhancing(fpimg)
        enhanced_image = enhancer.advancedEnhancing(enhanced_image)
        utils.saveImageToDatabase(label, enhanced_image)

        response = {'status': 'Saved successfully!'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/predict_with_db', methods=['POST'])
def predictWithDb():
    if request.method == 'POST':
        fp = request.files['fp1'].read()

        fpimg = np.frombuffer(fp, np.uint8)
        fpimg = cv2.imdecode(fpimg, cv2.IMREAD_GRAYSCALE)

        out1 = enhancer.basicEnhancing(fpimg)
        main_img = enhancer.advancedEnhancing(out1)
        all_db_imgs = utils.getAllImagesFromDatabase()

        pred_result = None
        try:
            pred_result = predictor.getPredictionDb(main_img, all_db_imgs)
            pred_result['best_pred']['accuracy'] = float("%.5f" %(pred_result['best_pred']['accuracy']))
        except Exception as e:
            print(e)
            return Response(response={'status': 'Predction Problem! Check image size, maybe.'}, status=503, mimetype="application/json")

        response_pickled = jsonpickle.encode(pred_result)
        return Response(response=response_pickled, status=200, mimetype="application/json")

#get all stored fingerprints database -> json
@app.route('/get_db')
def getDb():
    response = {'status': 'successful'}
    response['data'] = utils.getAllImagesFromDatabase()
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/test')
def test():
    result = ""
    for i in range(1, 5):
        img1 = cv2.imread(f'{i}.png', cv2.IMREAD_GRAYSCALE)
        for j in range(1,5):
            img2 = cv2.imread(f'{j}.png', cv2.IMREAD_GRAYSCALE)
            result +=  f"{i}-{j} --> {predictor.two_image_prediction(img1, img2) * 100} \n"
        result += "\n"
    print(result)
    return Response(response=result, status=200, mimetype="text/plain")

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large (Must be less than 2MB)', 413

@app.errorhandler(503)
def request_model_not_responding(error):
    return 'Model is Not responding', 503
