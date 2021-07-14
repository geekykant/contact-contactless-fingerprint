import tflit, cv2, numpy as np
from os.path import join, dirname

dirname = dirname(__file__)
model_path = join(dirname, '../model/model.tflite')
model = tflit.Model(model_path)

#Testing with all other database images
def getPredictionDb(main_img):
	main_img = np.array(main_img, dtype='uint8')

	#resize image into 220, 220 if not.
	if main_img.shape != (220, 220):
		main_img = cv2.resize(main_img, (220, 220))

	preds = model.predict(main_img.reshape((1,220,220,1)))[0]
	formatted_preds = list(map(lambda x: "%.3f" %(x), preds.tolist()))

	best_pred_idx = int(np.argmax(preds))
	print(preds[best_pred_idx])
	if(preds[best_pred_idx] < 0.5):
		return {"best_pred_pred": None, "best_pred_idx": None, "all_preds": formatted_preds}

	return {"best_pred_pred": preds[best_pred_idx]*100 , "best_pred_idx": best_pred_idx + 1, "all_preds": formatted_preds}
