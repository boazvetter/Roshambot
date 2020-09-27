'''
Robot hand plays a game of rock, paper scisssors with humans using a webcam input.


Boaz Vetter
VU Amsterdam 2020
'''
import numpy as np
import cv2

import serial
import time
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras.utils.generic_utils import CustomObjectScope


# Initialize serial connection with arduino and set baud rate
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=.1)

# Load model
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
	'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = tf.keras.models.load_model('rock-paper-scissors-model.h5',
    	custom_objects={'relu6': keras.applications.mobilenet.relu6})

previousTime = time.perf_counter()


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

def move_arm(prediction):
    print('Prediction: ', prediction)
    print(type(prediction))
    if prediction == 0:
        arduino.write("r".encode('utf-8'))
    elif prediction == 1:
        arduino.write("p".encode('utf-8'))
    elif prediction == 2:
        arduino.write("s".encode('utf-8'))
    pass    


# TODO Show webcam feed, overlay prediction
cap = cv2.VideoCapture(0)
ready_to_move = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    img = frame

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  cap.get(cv2.CAP_PROP_FPS)

    cv2.rectangle(frame, (int(width*0.25),int(height*0.25)), (int(width*0.75),int(height*0.75)), (0, 255, 0), 3)

    roi = frame[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.75)]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data = np.array(img)

    prediction = model.predict(np.array([img]), verbose = 1, use_multiprocessing = False)
    predicted_move = np.argmax(prediction)
    predicted_move = to_string(predicted_move)
    print(prediction, "Predicted: {}".format(predicted_move))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, predicted_move, (int(width*0.25),int(height*0.20)), 
    	font, 1.2, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow('Roshambot',frame)

    # Move the robot arm
    if(time.perf_counter() > (previousTime+3)):
        if (np.argmax(prediction) != 3):
            print("################################## Moving arm #############################")
            move_arm(np.argmax(prediction))
            previousTime = time.perf_counter()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# TODO implement robot RPS strategy

# TODO implement python-arduino serial interface



# TODO scoreboard

# TODO voice 

