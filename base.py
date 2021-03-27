import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

# print("Version: ", tf.__version__)
# print("Eager mode: ", tf.executing_eagerly())
# print("GPU: ", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.device('/cpu:0'):

	# model definition
	def build_model():
	    x1 = layers.Input(shape = (160, 160, 1))
	    x2 = layers.Input(shape = (160, 160, 1))

	    # share weights both inputs
	    inputs = layers.Input(shape = (160, 160, 1))
	    feature = layers.Conv2D(32, kernel_size = 3, activation = 'relu')(inputs)
	    feature = layers.MaxPooling2D(pool_size = 2)(feature)
	    feature = layers.Conv2D(64, kernel_size = 3, activation = 'relu')(feature)
	    feature = layers.MaxPooling2D(pool_size = 2)(feature)
	    feature = layers.Conv2D(128, kernel_size = 3, activation = 'relu')(feature)
	    feature = layers.MaxPooling2D(pool_size = 2)(feature)
	    feature_model = Model(inputs = inputs, outputs = feature)

	    # show feature model summary
	    # feature_model.summary()

	    # two feature models that sharing weights
	    x1_net = feature_model(x1)
	    x2_net = feature_model(x2)

	    # subtract features
	    net = layers.Subtract()([x1_net, x2_net])
	    net = layers.Conv2D(128, kernel_size = 3, activation = 'relu')(net)
	    net = layers.MaxPooling2D(pool_size = 2)(net)
	    net = layers.Flatten()(net)
	    net = layers.Dense(512, activation = 'relu')(net)
	    net = layers.Dense(1, activation = 'sigmoid')(net)
	    model = Model(inputs = [x1, x2], outputs = net)

	    # compile
	    model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr=1e-4), metrics = ['acc'])

	    # show summary
	    # model.summary()
	    return (model, feature_model)

	(model, feature_model) = build_model()
	# load model
	model_path = 'model/fp160.h5'
	model_feature_path = 'model/fp160_feature.h5'
	feature_model = tf.keras.models.load_model(model_feature_path, compile=False)
	model = tf.keras.models.load_model(model_path, compile=False)
	# feature_model.summary()
	# model.summary()

	#Contactless images testing (REAL PART)
	import cv2

	input_idx = 'L'
	db_idx = 'R'

	def two_image_prediction(input1_file, input2_file):
	    input1_img = cv2.imread(input1_file, cv2.IMREAD_GRAYSCALE)
	    input1_img = input1_img.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
	    input2_img = cv2.imread(input2_file, cv2.IMREAD_GRAYSCALE)
	    input2_img = input2_img.reshape((1, 160, 160, 1)).astype(np.float32) / 255.
	    pred_right = model.predict([input1_img, input2_img])

	    print("Prediction %.5f" %pred_right[0][0])

	    # plt.figure(figsize=(8, 4)).suptitle("Prediction %.5f" %pred_right[0][0])
	    # plt.subplot(1, 2, 1)
	    # plt.title('Input: %s' %input_idx)
	    # plt.imshow(input1_img.squeeze(), cmap='gray')
	    # plt.subplot(1, 2, 2)
	    # plt.title('O: %s' % (db_idx))
	    # plt.imshow(input2_img.squeeze(), cmap='gray')
	    # plt.show()

	#checking contactless i/p images one to one (1-1)
	for i in range(1,5):
	  for j in range(1,5):
	    two_image_prediction(f'{i}.png', f'{j}.png')