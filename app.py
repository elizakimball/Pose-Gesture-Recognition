import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import MultiPointHistoryClassifier

def main():   
    # For webcam input:
    cap = cv.VideoCapture(0)

    # Model load
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose 
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    multi_point_history_classifier = MultiPointHistoryClassifier()

    # Read labels and creates 2 lists for keypoint and point history labels. 
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
    with open(
            'model/multi_point_history_classifier/multi_point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        multi_point_history_classifier_labels = csv.reader(f)
        multi_point_history_classifier_labels = [
            row[0] for row in multi_point_history_classifier_labels
        ]

   # FPS Measurement (frames per second)
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history 
    history_length = 16
    point_history = deque(maxlen=history_length)
    multi_point_history = deque(maxlen=history_length)

    # Left wrist id history 
    left_wrist_id_history = deque(maxlen=history_length)
    multi_point_id_history = deque(maxlen=history_length)

    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) 
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation 
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB) 

        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True

        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())   
        
        # Counts the first few landmarks which exclude the leg landmarks
        landmark_count = 25 

        if results.pose_landmarks is not None:
            # pose_value holds x,y,z, and visibility for the 32 landmarks of each frame. 
            pose_values = getattr(results.pose_landmarks, "landmark")

            # Create blank frame list where each element in the list will hold a list with x and y values
            frame_list = []

            # landmark_value gives 1 of the 32 landmarks for each frame
            for i, landmark_value in enumerate(pose_values):
                x = getattr(landmark_value, 'x')
                y = getattr(landmark_value, 'y')
                frame_list.append([x,y])

            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, frame_list)

            # Landmark calculation. Landmark_list is a list for the 33 landmarks where each element is [x, y].
            landmark_list = calc_landmark_list(debug_image, frame_list)

            #Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
            pre_processed_multi_point_history_list = pre_process_multi_point_history(debug_image, multi_point_history)

            # Write to the dataset file 
            logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list, pre_processed_multi_point_history_list)

            # Pose classification
            pose_id = keypoint_classifier(pre_processed_landmark_list)
            point_history.append(landmark_list[16]) # for left wrist
            multi_point_history.append(landmark_list[:landmark_count]) # for most of body
            # Left wrist classification
            left_wrist_id = 0 # Default
            point_history_len = len(pre_processed_point_history_list)
            #print('Point History Length:', len(point_history))
            #print('Processed Point History Length:', point_history_len)
            if point_history_len == (history_length * 2):
                left_wrist_id = point_history_classifier(
                    pre_processed_point_history_list)
                
            # Full body classification
            full_body_id = 0 # Default
            multi_point_history_len = len(pre_processed_multi_point_history_list)
            #print('Multi Point History Length:', multi_point_history_len)
            #print('Processed multi:', multi_point_history_len)
            if multi_point_history_len == (history_length * landmark_count * 2):
                full_body_id = multi_point_history_classifier(
                    pre_processed_multi_point_history_list)

            # Calculates the left wrist IDs in the latest detection
            left_wrist_id_history.append(left_wrist_id)
            most_common_lw_id = Counter(
                left_wrist_id_history).most_common()
            
            # Calculates the full body IDs in the latest detection
            multi_point_id_history.append(full_body_id)
            most_common_multi_id = Counter(
                multi_point_id_history).most_common()            
            

            # Add classification to image
            image = draw_info_text(
                image,
                brect,
                keypoint_classifier_labels[pose_id],
                point_history_classifier_labels[most_common_lw_id[0][0]],
                multi_point_history_classifier_labels[most_common_multi_id[0][0]]
                )

            
        else:
            point_history.append([0, 0])
            default_coordinates = [0,0]
            multi_default_coordinates = [default_coordinates] * landmark_count
            multi_point_history.append(multi_default_coordinates)

        image = draw_info(image, fps, mode, number)
        
        # Screen reflection 
        cv.imshow('MediaPipe Pose', image)

    cap.release()
    cv.destroyAllWindows()
      
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    if key == 98:  # b
        mode = 3
    return number, mode

# With alterations this function will need to take in a 2D list in the place of second argument
def calc_bounding_rect(image, frame):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(frame):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# With alterations this function will need to take in a 2D list in the place of second argument
def calc_landmark_list(image, frame):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(frame):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)
        # landmark_z = landmark.z
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def pre_process_multi_point_history(image, multi_point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_multi_point_history = copy.deepcopy(multi_point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, multi_point in enumerate(temp_multi_point_history):
        # multi_point is a 2D list of multiple landmarks at a given time
        # Create another for loop to go through each landmark
        for landmark_id, landmark_point in enumerate(multi_point):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            multi_point[landmark_id][0] = (multi_point[landmark_id][0] - 
                                           base_x) / image_width
            multi_point[landmark_id][1] = (multi_point[landmark_id][1] - 
                                           base_y) / image_height
            
        # Convert to a one-dimensional list
        multi_point = list(
        itertools.chain.from_iterable(multi_point))
        temp_multi_point_history[index] = multi_point

    # Convert to a one-dimensional list
    temp_multi_point_history = list(
        itertools.chain.from_iterable(temp_multi_point_history))
    
    return temp_multi_point_history


def logging_csv(number, mode, landmark_list, point_history_list, multi_point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    if mode == 3 and (0 <= number <= 9):
        csv_path = 'model/multi_point_history_classifier/multi_point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *multi_point_history_list])
    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, pose_class_text, left_wrist_text, multi_point_text):
    if pose_class_text != "":
        cv.putText(image, "Classification:" + pose_class_text, (10, 70),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv.LINE_AA)      
    if left_wrist_text != "":
        cv.putText(image, "Left Wrist Motion:" + left_wrist_text, (10, 105),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
    if multi_point_text != "":
        cv.putText(image, "Full Body Motion:" + multi_point_text, (10, 140),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
    return image


# Adds circles on video to represent each landmark. 
def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


# Adds logging information to video
def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History', 'Logging Multi Point History']
    if 1 <= mode <= 3:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, image.shape[0] - 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, image.shape[0] - 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
