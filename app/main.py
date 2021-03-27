import os, sys
from predictor import two_image_prediction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from flask import Flask, render_template, request, redirect, url_for, Response
import cv2, jsonpickle, numpy as np

# start flask
app = Flask(__name__, template_folder='templates')

# render default webpage
@app.route('/')
def home():
    return render_template('index.html')

# when the post method detect, then redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        # fp1_image = request.files.get('fp1', '')
        # fp2_image = request.files.get('fp2', '')

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
            prediction = two_image_prediction(fp1img, fp2img) * 100
            prediction_result = float("%.5f" %prediction)
        except Exception as e:
            print(e)
            return Response(response={'status': 'Predction Error'}, status=503, mimetype="application/json")

        response = {'accuracy': prediction_result, 'status': 'up & running'}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype="application/json")

@app.errorhandler(413)
def request_entity_too_large(error):
    return 'File Too Large', 413

@app.errorhandler(503)
def request_model_not_responding(error):
    return 'Model is Not responding', 503

if __name__=="__main__":
    app.run(debug=True)
