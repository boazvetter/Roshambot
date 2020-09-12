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
from keras.utils import to_categorical

target_classes = {
	'rock':0, 
	'paper':1, 
	'scissors':2, 
	'none':3}
imagepath = 'images' # Name of imagefolder

def to_numeric(label):
	if label == 'rock':
		return 0
	elif label == 'paper':
		return 1
	elif label == 'scissors':
		return 2
	elif label == 'none':
		return 3
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
		#print(filenames)
		f.extend(filenames)
		f = sorted_alphanumeric(f)
		f = f[:10]
	for file in f:
		imgpath = target_folder+'{}{}'.format('/',f[0])
		img = cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (227,227))
		dataset.append([img, target])
		dataset_pathlist.append(imgpath)

data, labels = zip(*dataset)
data = np.array(data)
labels = list(map(to_numeric, labels))
labels = to_categorical(labels)

# Load the last model that we trained
model = tf.keras.models.load_model('rock-paper-scissors-model.h5')

# Predict on some data
predictions = model.predict(np.array(data), verbose = 1, use_multiprocessing = False)
move_code = np.argmax(predictions[0])
move_name = to_numeric(move_code)

#print(predictions[0], "Predicted: {}".format(move_name))

for i, prediction in enumerate(predictions):
	print(dataset_pathlist[i])
	print(prediction, "Predicted: {}".format(move_name))
	print("---------------------------------------------")