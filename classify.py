# USAGE
# python classify.py --model weather.model --labelbin weather.pickle

# import the necessary packages
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())

# load the trained convolutional neural network and the multi-label
# binarizer
model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())

vs = cv2.VideoCapture("mygeneratedvideo.avi")
while True:
	(grabbed, image) = vs.read()
		
	if not grabbed:
		break

	output = imutils.resize(image, width=400)
		 
	# pre-process the image for classification
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# classify the input image then find the indexes of the two class
	# labels with the *largest* probability
	proba = model.predict(image)[0]
	idxs = np.argsort(proba)[::-1][:2]

	# loop over the indexes of the high confidence class labels
	for (i, j) in enumerate(idxs):
		# build the label and draw the label on the image
		label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
		cv2.putText(output, label, (10, (i * 30) + 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		break

	# show the output image
	cv2.imshow("Output", output)
	time.sleep(1)  # Sleep for 3 seconds

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break