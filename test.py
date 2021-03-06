'''
Test script to check the performance of the trained model
network to four image classes (Rock, Paper, Scissors, None)


Boaz Vetter
VU Amsterdam 2020
'''
import os
from os import walk
import re

import cv2
import numpy as np
import tensorflow as tf
import keras

from keras.utils import to_categorical
from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
	'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = tf.keras.models.load_model('rock-paper-scissors-model.h5',
    	custom_objects={'relu6': keras.applications.mobilenet.relu6})


target_classes = {
	'rock':0, 
	'paper':1, 
	'scissors':2, 
	'none':3}
imagepath = 'images' # Name of imagefolder

def to_numeric(prediction):
	if prediction == 'rock':
		return 0
	elif prediction == 'paper':
		return 1
	elif prediction == 'scissors':
		return 2
	elif prediction == 'none':
		return 3
	pass

def to_string(prediction):
	if prediction == 0:
		return 'rock'
	elif prediction == 1:
		return 'paper'
	elif prediction == 2:
		return 'scissors'
	elif prediction == 3:
		return 'none'
	pass	

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# Prepare images to predict on
imagepath = 'images' # Name of dataset folder
path = os.getcwd()
dataset = []
target_path = os.path.join(path, imagepath)
#sample_image = target_path+'{}{}'.format('/','5.jpg')
# img = cv2.imread(sample_image)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (227,227))

dataset_pathlist = []
for target in target_classes:
	target_folder = os.path.join(target_path, target)
	f = []

	for (dirpath, dirnames, filenames) in walk(target_folder):
		f.extend(filenames)
		f = sorted_alphanumeric(f)
		f = f[:10]
	for file in f:
		imgpath = target_folder+'{}{}'.format('/',file)
		img = cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224,224))
		dataset.append([img, target])
		dataset_pathlist.append(imgpath)

data, labels = zip(*dataset)
data = np.array(data)
labels = list(map(to_numeric, labels))
labels = to_categorical(labels)

# Load the last model that we trained
#model = tf.keras.models.load_model('rock-paper-scissors-model.h5')

# Predict on some data
predictions = model.predict(np.array(data), verbose = 1, use_multiprocessing = False)

print(dataset_pathlist)

for i, prediction in enumerate(predictions):
	print(dataset_pathlist[i])
	predicted_move = np.argmax(prediction)
	predicted_move = to_string(predicted_move)
	print(prediction, "Predicted: {}".format(predicted_move))
	print("---------------------------------------------")