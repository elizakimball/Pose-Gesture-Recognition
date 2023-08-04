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

class PoseDetection:
    def __init__(self):

        # Model load
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose 
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()
        self.multi_point_history_classifier = MultiPointHistoryClassifier()

        # Read labels and creates 2 lists for keypoint and point history labels. 
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]
        with open(
                'model/multi_point_history_classifier/multi_point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.multi_point_history_classifier_labels = csv.reader(f)
            self.multi_point_history_classifier_labels = [
                row[0] for row in self.multi_point_history_classifier_labels
            ]

        # FPS Measurement (frames per second)
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history 
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.multi_point_history = deque(maxlen=self.history_length)

        # Left wrist id history 
        self.left_wrist_id_history = deque(maxlen=self.history_length)
        self.multi_point_id_history = deque(maxlen=self.history_length)

        self.mode = 0

        # Create list for consecutive times a label id is repeated
        self.threshold = 20

        # Initialize multipoint id list
        self.most_common_multi_id = []

        # Counts the first few landmarks which exclude the leg landmarks
        self.landmark_count = 25 

    # Set video capture input
    def set_video_input(self, camera_input):
        return camera_input

    def run_pose_detection(self, camera_input=0):
        # For webcam_input:
        self.cap = cv.VideoCapture(camera_input) # 0 for Radhik'a webcam, 1 for my computer camera, 'gesture.mov' for imported video
        
        self.consecutive_count = 0
        while True:
            self.fps = self.cap.get(cv.CAP_PROP_FPS)
            #self.fps = self.cvFpsCalc.get()

            # Process Key (ESC: end) 
            self.key = cv.waitKey(10)
            if self.key == 27:  # ESC
                break
            if self.key == 97: # a
                self.user_input = input("Enter new multipoint label:")
                self.add_multipoint_label(self.user_input)
            self.number, self.mode = self.select_mode(self.key, self.mode)

            # Camera capture 
            ret, self.image = self.cap.read()
            if not ret:
                break
            self.image = cv.flip(self.image, 1)  # Mirror display
            self.debug_image = copy.deepcopy(self.image)

            # Detection implementation 
            self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB) 

            self.image.flags.writeable = False
            results = self.pose.process(self.image)
            self.image.flags.writeable = True

            self.image = cv.cvtColor(self.image, cv.COLOR_RGB2BGR)

            # Draw the pose annotation on the image.
            self.mp_drawing.draw_landmarks(
                self.image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = self.mp_drawing_styles.get_default_pose_landmarks_style())   

            # Create variable for previous multipoint id
            if not self.most_common_multi_id:
                previous_multipoint_id = -1
            else:
                previous_multipoint_id = self.most_common_multi_id[0][0]

            # Initialize variable 
            self.consecutive_multipoint_id = 0

            if results.pose_landmarks is not None:
                # pose_value holds x,y,z, and visibility for the 32 landmarks of each frame. 
                pose_values = getattr(results.pose_landmarks, "landmark")

                # Create blank frame list where each element in the list will hold a list with x and y values
                self.frame_list = []

                # landmark_value gives 1 of the 32 landmarks for each frame
                for i, landmark_value in enumerate(pose_values):
                    x = getattr(landmark_value, 'x')
                    y = getattr(landmark_value, 'y')
                    self.frame_list.append([x,y])

                # Bounding box calculation
                self.brect = self.calc_bounding_rect(self.debug_image, self.frame_list)

                # Landmark calculation. Landmark_list is a list for the 33 landmarks where each element is [x, y].
                self.landmark_list = self.calc_landmark_list(self.debug_image, self.frame_list)

                #Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.pre_process_landmark(self.landmark_list)
                pre_processed_point_history_list = self.pre_process_point_history(self.debug_image, self.point_history)
                pre_processed_multi_point_history_list = self.pre_process_multi_point_history(self.debug_image, self.multi_point_history)

                # Write to the dataset file 
                self.logging_csv(self.number, self.mode, pre_processed_landmark_list, pre_processed_point_history_list, pre_processed_multi_point_history_list)
                # Pose classification
                self.keypoint_classifier = KeyPointClassifier()
                self.pose_id = self.keypoint_classifier(pre_processed_landmark_list)
                self.point_history.append(self.landmark_list[16]) # for left wrist
                self.multi_point_history.append(self.landmark_list[:self.landmark_count]) # for most of body
                # Left wrist classification
                left_wrist_id = 0 # Default
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    left_wrist_id = self.point_history_classifier(
                        pre_processed_point_history_list)
                    
                # Full body classification
                full_body_id = 0 # Default
                multi_point_history_len = len(pre_processed_multi_point_history_list)
                if multi_point_history_len == (self.history_length * self.landmark_count * 2):
                    full_body_id = self.multi_point_history_classifier(
                        pre_processed_multi_point_history_list)

                # Calculates the left wrist IDs in the latest detection
                self.left_wrist_id_history.append(left_wrist_id)
                most_common_lw_id = Counter(
                    self.left_wrist_id_history).most_common()
                
                # Calculates the full body IDs in the latest detection
                self.multi_point_id_history.append(full_body_id)
                self.most_common_multi_id = Counter(
                    self.multi_point_id_history).most_common()        

                if self.most_common_multi_id == 4:
                    print("move forward")
                # Ensure multipoint id lasts for longer    
                if self.most_common_multi_id[0][0] == previous_multipoint_id:
                    self.consecutive_count += 1
                    if self.consecutive_count >= self.threshold:
                        self.consecutive_multipoint_id = self.most_common_multi_id[0][0]
                    else:
                        self.consecutive_multipoint_id = 0
                else:
                    self.consecutive_count = 0
                

                # Add classification to image
                self.image = self.draw_info_text(
                    self.image,
                    self.brect,
                    self.keypoint_classifier_labels[self.pose_id],
                    self.consecutive_count,
                    self.multi_point_history_classifier_labels[self.consecutive_multipoint_id]
                    )
                
            else:
                self.point_history.append([0, 0])
                default_coordinates = [0,0]
                multi_default_coordinates = [default_coordinates] * self.landmark_count
                self.multi_point_history.append(multi_default_coordinates)

            self.image = self.draw_info(self.image, self.fps, self.mode, self.number)

            # Screen reflection 
            cv.imshow('MediaPipe Pose', self.image)

        self.cap.release()
        cv.destroyAllWindows()
        
    def add_multipoint_label(self, label):
        csv_path = 'model/multi_point_history_classifier/multi_point_history_classifier_label.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label])
        return

    def select_mode(self, key, mode):
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
    def calc_bounding_rect(self, image, frame):
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
    def calc_landmark_list(self, image, frame):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(frame):
            landmark_x = min(int(landmark[0] * image_width), image_width - 1)
            landmark_y = min(int(landmark[1] * image_height), image_height - 1)
            # landmark_z = landmark.z
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
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


    def pre_process_point_history(self, image, point_history):
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


    def pre_process_multi_point_history(self, image, multi_point_history):
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


    def logging_csv(self, number, mode, landmark_list, point_history_list, multi_point_history_list):
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


    def draw_info_text(self, image, brect, pose_class_text, consecutive_count, multi_point_text):
        if pose_class_text != "":
            cv.putText(image, "Classification:" + pose_class_text, (10, 70),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                        cv.LINE_AA)      
        if consecutive_count != "":
            cv.putText(image, "Number of repetitions for current classification:" + str(consecutive_count), (10, 105),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv.LINE_AA)
        if multi_point_text != "":
            cv.putText(image, "Full Body Motion:" + multi_point_text, (10, 140),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv.LINE_AA)
        return image


    # Adds circles on video to represent each landmark. 
    def draw_point_history(self, image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                        (152, 251, 152), 2)

        return image


    # Adds logging information to video
    def draw_info(self, image, fps, mode, number):
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
    pose_detection = PoseDetection()
    #camera_input = pose_detection.set_video_input('gesture.mov')
    pose_detection.run_pose_detection()
