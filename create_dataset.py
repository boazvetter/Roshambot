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
import copy

try:
	label_name = sys.argv[1]
	num_pictures = sys.argv[2]
except:
	print("Please try again!")
	print("Syntax: python3 create_dataset.py GESTURE_TYPE AMOUNT_OF_PICTURES")
	sys.exit()	


path = os.getcwd()
img_dir = 'images'
label_path = os.path.join(img_dir, label_name)

# Make directories
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

print("Press spacebar to create dataset")
filenumber = 0
collect = False

while(True):
	ret, frame = cap.read()

	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )

	img = copy.deepcopy(frame)
	cv2.rectangle(img, (int(width*0.25),int(height*0.25)), (int(width*0.75),int(height*0.75)), (0, 255, 0), 3)
	cv2.imshow('Dataset creator', img)

	if filenumber == int(num_pictures):
		print("Done!")
		break	

	k = cv2.waitKey(1)
	if k == 32:
		print("Collecting data")
		collect = True
		
	elif k == ord('q'):
		print("Exiting..")
		break

	if collect == True:
		img_path = os.path.join(path, img_dir, label_name)		
		roi = frame[int(height*0.25):int(height*0.75), int(width*0.25):int(width*0.75)]
		cv2.imwrite(img_path+'/'+str(+filenumber)+'.jpg', roi)
		filenumber += 1		


cap.release()
cv2.destroyAllWindows()

