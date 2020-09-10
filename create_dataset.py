''' 
This commandline script takes two input arguments 
Usage: python3 create_dataset.py GESTURE PICTURES
Where folder should be 'Rock, Paper, Scissors or None'
And Pictures should be the amount of training pictures made for the dataset


Boaz Vetter 2020
VU Amsterdam
'''
import sys
import cv2
import os


try:
	label_name = sys.argv[1]
	num_pictures = sys.argv[2]
except:
	print("Please try again!")
	print("Syntax: python3 create_dataset.py GESTURE_TYPE AMOUNT_OF_PICTURES")
	sys.exit()	


path = os.getcwd()
img_path = 'images'
label_path = os.path.join(path, img_path, label_name)

# Make directory
try:
    os.mkdir('images')
except FileExistsError:
    pass
try:
    os.mkdir(label_path)
except FileExistsError:
    pass    

# Show webcam feed

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()

	cv2.imshow('Dataset creator', frame)
	if cv2.waitkey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


# Take pictures with keystroke



# Save as argv childfolder 