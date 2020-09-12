'''
This is a training script to fine-tune the weights of an existing
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
labels = list(map(to_numeric, labels))
labels = to_categorical(labels)

# Train model



 













# # create the base pre-trained model
# base_model = InceptionV3(weights='imagenet', include_top=False)

# # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(3, activation='softmax')(x)

# # this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# # train the model on the new data for a few epochs
# model.fit(...)

# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.

# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from tensorflow.keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit(...)