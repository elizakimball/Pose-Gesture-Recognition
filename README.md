This repository estimates full body poses using the Python version of MediaPipe. It uses the pose detection model from Mediapipe to identify the 32 landmarks on a full body. The keypoint model is responsible for classifying static poses while the point history and multipoint history classifies left hand motion and full body motion, respectively. 

This code has been adapted from https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/app.py. 

The current classifications for full body motion (as seen in multipoint history) are default, turn left, turn right, move forward, and slow down. The multipoint history model only tracks motion data from the first 25 landmarks which include all body landmarks from Mediapipe’s pose detection model except for the leg landmarks. The point history model tracks motion data for the left wrist, but it can be easily adapted to accomodate any  landmark from the 32 full body landmarks.

Push one of the following lower-case letters after running app.py to alter the csv files:
a -> This will prompt the user for a new multipoint classifier label that will be added to multi_point_history_classifier_label.csv
The following 4 modes are used for logging new data:
n -> (mode 0) This key returns the model to mode 0 where it cannot log any new data.
k -> (mode 1) While in mode 1 and pushing a number between 0 and 9 (inclusive), stationary image data will be recorded for keypoints.
h -> (mode 2) While in mode 2 and pushing a number between 0 and 9 (inclusive), data will be recorded for left wrist motion.
b -> (mode 3) While in mode 3 and pushing a number between 0 and 9 (inclusive), data will be recorded for full motion.
