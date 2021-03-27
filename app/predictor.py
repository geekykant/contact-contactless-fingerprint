import tflit, cv2, numpy as np
import os

dirname = os.path.dirname(__file__)
model_path = os.path.join(dirname, '../model/model.tflite')
model = tflit.Model(model_path)

#Contactless images testing (REAL PART)
def two_image_prediction(np_img1, np_img2):
	input1_img = np_img1.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
	input2_img = np_img2.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
	pred_right = model.predict([input1_img, input2_img])

	return pred_right[0][0]
