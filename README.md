# Roshambot

An AI that will play Roshambo, also known as Rock-Paper-Scissors with humans. Through webcam input, the AI recognizes what hand gesture is played and strategizes accordingly. The strategy is then sent over serial communication to an Arduino microcontroller which is connected to a robot arm.

Video: https://www.youtube.com/watch?v=H3N9UBhsnaM

# Installation

This project runs on Python 3.7 and uses Tensorflow+Keras to recognize hands gestures.
Setup a virtual environment and install all dependencies:
```sh
pip install -r requirements.txt
```

# Use

Create_dataset.py: Usage: python3 create_dataset.py GESTURE PICTURES
This is used to create a dataset for the neural network to learn gestures from. At least 100 training examples per gesture is advised.

Train.py: Fine-tunes the weights of a popular convolutional neural network architecture (MobileNet) on our dataset and outputs a model.

Test.py: Inspects the results to see if we're happy with the model we created.

Game.py: Shows a webcam feed of the user with predicted actions, and controls the robotic arm accordingly by sending movements over serial communication to an Arduino microcontroller.
