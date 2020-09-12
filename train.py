'''
Training script to fine-tune the weights of an existing
network to four image classes (Rock, Paper, Scissors, None)


Boaz Vetter
VU Amsterdam 2020
'''
import os
from os import walk
import numpy as np
import cv2
import re

from keras.utils import to_categorical

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

target_classes = {
	'rock':0, 
	'paper':1, 
	'scissors':2, 
	'none':3}
imagepath = 'images' # Name of imagefolder

class_length = len(target_classes)

# Read in all images a numpy array using opencv
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def to_numeric(label):
	return target_classes[label]

path = os.getcwd()
dataset = []
target_path = os.path.join(path, imagepath)

for target in target_classes:
	target_folder = os.path.join(target_path, target)
	f = []
	for (dirpath, dirnames, filenames) in walk(target_folder):
		#print(filenames)
		f.extend(filenames)
		f = sorted_alphanumeric(f)
		print(target_folder+'{}{}'.format('/',f[0]))
	for file in f:
		imgpath = target_folder+'{}{}'.format('/',f[0])
		img = cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (227,227))
		dataset.append([img, target])

# Create one-hot encoding to avoid errors with label length
data, labels = zip(*dataset)
data = np.array(data)
labels = list(map(to_numeric, labels))
labels = to_categorical(labels)

################## Train model ####################

base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Add a logistic layer with our rock,paper,scissors classes
predictions = Dense(class_length, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(
	optimizer=rmsprop(lr=0.001), 
	loss='categorical_crossentropy',
	metrics=['accuracy'])

model.fit(data, labels, epochs=5)
# visualize layer names and layer indices to see how many layers we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# Recompile the model for these modifications to take effect
from tensorflow.keras.optimizers import SGD
model.compile(
	optimizer=SGD(lr=0.0001, momentum=0.9), 
	loss='categorical_crossentropy',
	metrics=['accuracy'])

# Train the model againn
model.fit(data, labels, epochs=5)


# Save the model
model.save('rock-paper-scissors-model.h5')