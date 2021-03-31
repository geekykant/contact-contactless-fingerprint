import tflit, cv2, numpy as np
from os.path import join, dirname

dirname = dirname(__file__)
model_path = join(dirname, '../model/model.tflite')
model = tflit.Model(model_path)

#Contactless images testing (REAL PART)
def two_image_prediction(np_img1, np_img2):
	input1_img = np_img1.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
	input2_img = np_img2.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
	pred_right = model.predict([input1_img, input2_img])

	return pred_right[0][0]

#Testing with all other database images
def getPredictionDb(main_img, all_db_imgs):
	best_pred = 0.00
	best_html_id = "db_null"
	matched_person = "null"

	main_img = np.array(main_img, dtype='uint8')

	#resize image into 160 x 160 if not.
	if main_img.shape != (160, 160):
		main_img = cv2.resize(main_img, (160, 160))

	for file in all_db_imgs:
		file_url = join(dirname, file['url'][1:])
		db_img = cv2.imread(file_url, cv2.IMREAD_GRAYSCALE)
		db_img = np.array(db_img, dtype='uint8')

		if db_img.shape != (160, 160):
			db_img = cv2.resize(db_img, (160, 160))

		pred = two_image_prediction(main_img, db_img)
		# print(pred)
		if pred > best_pred:
			best_pred = pred
			matched_person = file['label']
			best_html_id = f"db_{matched_person}"

	result = {"html_id": best_html_id, "accuracy": best_pred * 100, "matched_person": matched_person}
	return result
