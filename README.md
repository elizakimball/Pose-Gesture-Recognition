This repository estimates full body poses using the Python version of MediaPipe. The keypoint model is responsible for classifying static poses while the point history and multipoint history classifies left hand and full body motion, respectively. It uses the pose detection model from Mediapipe to identify the 32 landmarks on a full body. 

This code has been adapted from https://github.com/kinivi/hand-gesture-recognition-mediapipe/blob/main/app.py. 

The current classifications for full body motion (as seen in multipoint history) are stop/no motion, turn left, turn right, move forward, and slow down. Multi_point_history.csv only tracks data from the first 25 landmarks which include all body landmarks from Mediapipe’s pose detection model except for the leg landmarks. 

