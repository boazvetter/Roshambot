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
from tensorflow.keras.layers import Input

target_classes = {
	'rock':0, 
	'paper':1, 
	'scissors':2, 
	'none':3}
imagepath = 'images' # Name of imagefolder

CLASS_LENGTH = len(target_classes)

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
		f.extend(filenames)
		f = sorted_alphanumeric(f)
	for file in f:
		imgpath = target_folder+'{}{}'.format('/',file)
		img = cv2.imread(imgpath)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (224,224))
		dataset.append([img, target])

# Create one-hot encoding to avoid errors with label length
data, labels = zip(*dataset)
data = np.array(data)
labels = list(map(to_numeric, labels))
labels = to_categorical(labels)

################## Train model ####################


import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam

mobile = keras.applications.mobilenet.MobileNet()

base_model=MobileNet(input_shape=(224, 224, 3), weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation

model=Model(inputs=base_model.input,outputs=preds)

# for i,layer in enumerate(model.layers):
#   print(i,layer.name)

for layer in model.layers:
    layer.trainable=False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True 

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(np.array(data), np.array(labels), epochs=5)

# Save the model
model.save('rock-paper-scissors-model.h5')