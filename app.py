from flask import Flask, render_template, Response, request, redirect, url_for, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import mediapipe as mp
import math
import time
import json
import os
from datetime import datetime
import numpy as np
from collections import defaultdict

app = Flask(__name__)
CORS(app)
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Global variables for pose tracking and messaging
pose_state = 1
continue_session = True
display_message = "Initializing pose tracking..."
accuracy_message = "Accuracy: 0%"
state_start_time = None
hold_duration = 2  # seconds to hold each state
state_max_accuracies = [0] * 5  # Max accuracy for each state (default size)

# Session management variables
session_mode = False
session_poses = []         # List of poses for the session
session_index = 0          # Which pose in the session we are on
session_results = {}       # Pose name -> average accuracy
current_pose_accuracies = []  # Accumulate accuracy values for current pose
current_pose = "Tree Pose" # Default pose (will be set by query parameter)

# Session history storage
session_history = []

# -----------------------
# Helper Functions
# -----------------------
def detectPose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), landmark.z * width))
    return output_image, landmarks

def calculateAngle(a, b, c):
    x1, y1, _ = a
    x2, y2, _ = b
    x3, y3, _ = c
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle 

def calculate_accuracy(current_angles, ideal_angles):
    total_accuracy = 0
    for joint, ideal in ideal_angles.items():
        current = current_angles.get(joint, 0)
        joint_accuracy = max(0, 100 - abs(ideal - current))
        total_accuracy += joint_accuracy
    return total_accuracy / len(ideal_angles)

def draw_hold_timer(output_image):
    global state_start_time, hold_duration
    if state_start_time is not None:
        elapsed = time.time() - state_start_time
        remaining = hold_duration - elapsed
        if remaining < 0:
            remaining = 0
        cv2.putText(output_image, f"Hold: {remaining:.1f}s", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    return output_image

# -----------------------
# Pose Sequence Functions
# -----------------------
def tree_pose_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies, state_max_accuracies
    
    global state_max_accuracies
    if 'state_max_accuracies' not in globals() or len(state_max_accuracies) != 5:
        state_max_accuracies = [0] * 5

    current_angles = {
        "left_knee": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "right_knee": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        "left_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        "right_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    }

    if pose_state == 1:
        ideal_angles = {"left_knee": 180, "right_knee": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 2:
        left_lifted = current_angles["left_knee"] < 60 and 165 < current_angles["right_knee"] < 195
        right_lifted = current_angles["right_knee"] < 60 and 165 < current_angles["left_knee"] < 195
        if left_lifted:
            ideal_angles = {"left_knee": 90, "right_knee": 180, "left_shoulder": 20, "right_shoulder": 20}
        elif right_lifted:
            ideal_angles = {"left_knee": 180, "right_knee": 90, "left_shoulder": 20, "right_shoulder": 20}
        else:
            ideal_angles = {"left_knee": 180, "right_knee": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 3:
        left_lifted = current_angles["left_knee"] < 60 and 165 < current_angles["right_knee"] < 195
        if left_lifted:
            ideal_angles = {"left_knee": 90, "right_knee": 180, "left_shoulder": 180, "right_shoulder": 180}
        else:
            ideal_angles = {"left_knee": 180, "right_knee": 90, "left_shoulder": 180, "right_shoulder": 180}
    elif pose_state == 4:
        left_lifted = current_angles["left_knee"] < 60 and 165 < current_angles["right_knee"] < 195
        if left_lifted:
            ideal_angles = {"left_knee": 90, "right_knee": 180, "left_shoulder": 20, "right_shoulder": 20}
        else:
            ideal_angles = {"left_knee": 180, "right_knee": 90, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 5:
        ideal_angles = {"left_knee": 180, "right_knee": 180, "left_shoulder": 20, "right_shoulder": 20}
    else:
        ideal_angles = {"left_knee": 180, "right_knee": 180, "left_shoulder": 20, "right_shoulder": 20}

    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if 1 <= pose_state <= 5:
        state_index = pose_state - 1
        state_max_accuracies[state_index] = max(state_max_accuracies[state_index], accuracy)

    if pose_state == 1:
        display_message = "Stand still."
        if (165 < current_angles["left_knee"] < 195 and 
            165 < current_angles["right_knee"] < 195 and 
            0 < current_angles["left_shoulder"] < 40 and 
            0 < current_angles["right_shoulder"] < 40):
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Lift one leg."
        left_lifted = (current_angles["left_knee"] < 60 and 165 < current_angles["right_knee"] < 195)
        right_lifted = (current_angles["right_knee"] < 60 and 165 < current_angles["left_knee"] < 195)
        if left_lifted or right_lifted:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 3
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 3:
        display_message = "Raise your hands and join them."
        if 170 < current_angles["left_shoulder"] < 190 and 170 < current_angles["right_shoulder"] < 190:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 4
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 4:
        display_message = "Lower your hands."
        left_lifted = (current_angles["left_knee"] < 60 and 165 < current_angles["right_knee"] < 195)
        right_lifted = (current_angles["right_knee"] < 60 and 165 < current_angles["left_knee"] < 195)
        if (15 < current_angles["left_shoulder"] < 40 and 
            15 < current_angles["right_shoulder"] < 40 and 
            (left_lifted or right_lifted)):
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 5
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 5:
        display_message = "Lower your leg to complete the pose."
        if (165 < current_angles["left_knee"] < 195 and 
            165 < current_angles["right_knee"] < 195 and 
            15 < current_angles["left_shoulder"] < 40 and 
            15 < current_angles["right_shoulder"] < 40):
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                overall_accuracy = sum(state_max_accuracies) / 5
                display_message = f"Tree Pose Completed! Overall Accuracy: {overall_accuracy:.2f}%"
                
                # Save the pose result
                session_data = {
                    'timestamp': datetime.now().isoformat(),
                    'poses': ['Tree Pose'],
                    'accuracies': {'Tree Pose': overall_accuracy},
                    'duration': time.time() - state_start_time,
                    'medal': get_medal_for_accuracy(overall_accuracy)
                }
                session_history.append(session_data)
                print(f"Session saved: {session_data}")  # Debug print
                
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def warrior1_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies, state_max_accuracies
    
    global state_max_accuracies
    if 'state_max_accuracies' not in globals() or len(state_max_accuracies) != 3:
        state_max_accuracies = [0] * 3

    current_angles = {
        "front_knee": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "back_leg": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        "left_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        "right_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    }

    if pose_state == 1:
        ideal_angles = {"front_knee": 180, "back_leg": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 2:
        ideal_angles = {"front_knee": 90, "back_leg": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 3:
        ideal_angles = {"front_knee": 90, "back_leg": 180, "left_shoulder": 180, "right_shoulder": 180}
    else:
        ideal_angles = {"front_knee": 90, "back_leg": 180, "left_shoulder": 180, "right_shoulder": 180}

    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if 1 <= pose_state <= 3:
        state_index = pose_state - 1
        state_max_accuracies[state_index] = max(state_max_accuracies[state_index], accuracy)

    if pose_state == 1:
        display_message = "Stand with feet apart."
        if current_angles["back_leg"] > 170 and current_angles["front_knee"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Bend your front knee to 90°."
        if 80 < current_angles["front_knee"] < 100 and current_angles["back_leg"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 3
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 3:
        display_message = "Raise your arms vertically."
        if 170 < current_angles["left_shoulder"] < 190 and 170 < current_angles["right_shoulder"] < 190:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                overall_accuracy = sum(state_max_accuracies) / 3
                display_message = f"Warrior 1 Pose Completed! Overall Accuracy: {overall_accuracy:.2f}%"
                
                # Save the pose result
                session_data = {
                    'timestamp': datetime.now().isoformat(),
                    'poses': ['Warrior 1'],
                    'accuracies': {'Warrior 1': overall_accuracy},
                    'duration': time.time() - state_start_time,
                    'medal': get_medal_for_accuracy(overall_accuracy)
                }
                session_history.append(session_data)
                
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def warrior2_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies, state_max_accuracies
    
    global state_max_accuracies
    if 'state_max_accuracies' not in globals() or len(state_max_accuracies) != 3:
        state_max_accuracies = [0] * 3

    current_angles = {
        "front_knee": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
        "back_leg": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "left_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        "right_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    }

    if pose_state == 1:
        ideal_angles = {"front_knee": 180, "back_leg": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 2:
        ideal_angles = {"front_knee": 90, "back_leg": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 3:
        ideal_angles = {"front_knee": 90, "back_leg": 180, "left_shoulder": 180, "right_shoulder": 180}
    else:
        ideal_angles = {"front_knee": 90, "back_leg": 180, "left_shoulder": 180, "right_shoulder": 180}

    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if 1 <= pose_state <= 3:
        state_index = pose_state - 1
        state_max_accuracies[state_index] = max(state_max_accuracies[state_index], accuracy)

    if pose_state == 1:
        display_message = "Stand with feet apart."
        if current_angles["back_leg"] > 170 and current_angles["front_knee"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Bend your front knee to 90°."
        if 80 < current_angles["front_knee"] < 100 and current_angles["back_leg"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 3
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 3:
        display_message = "Extend your arms outward."
        if 170 < current_angles["left_shoulder"] < 190 and 170 < current_angles["right_shoulder"] < 190:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                overall_accuracy = sum(state_max_accuracies) / 3
                display_message = f"Warrior 2 Pose Completed! Overall Accuracy: {overall_accuracy:.2f}%"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def triangle_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies, state_max_accuracies
    
    global state_max_accuracies
    if 'state_max_accuracies' not in globals() or len(state_max_accuracies) != 2:
        state_max_accuracies = [0] * 2

    current_angles = {
        "side_stretch": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]),
        "leg": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "left_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        "right_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    }

    if pose_state == 1:
        ideal_angles = {"side_stretch": 180, "leg": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 2:
        ideal_angles = {"side_stretch": 90, "leg": 180, "left_shoulder": 20, "right_shoulder": 20}
    else:
        ideal_angles = {"side_stretch": 90, "leg": 180, "left_shoulder": 20, "right_shoulder": 20}

    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if 1 <= pose_state <= 2:
        state_index = pose_state - 1
        state_max_accuracies[state_index] = max(state_max_accuracies[state_index], accuracy)

    if pose_state == 1:
        display_message = "Stand with feet apart."
        if current_angles["leg"] > 170 and current_angles["side_stretch"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Bend forward and reach toward your foot."
        if 80 < current_angles["side_stretch"] < 100 and current_angles["leg"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                overall_accuracy = sum(state_max_accuracies) / 2
                display_message = f"Triangle Pose Completed! Overall Accuracy: {overall_accuracy:.2f}%"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def lord_of_dance_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies, state_max_accuracies
    
    global state_max_accuracies
    if 'state_max_accuracies' not in globals() or len(state_max_accuracies) != 2:
        state_max_accuracies = [0] * 2

    current_angles = {
        "leg_raise": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        "left_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        "right_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    }

    if pose_state == 1:
        ideal_angles = {"leg_raise": 180, "left_shoulder": 20, "right_shoulder": 20}
    elif pose_state == 2:
        ideal_angles = {"leg_raise": 90, "left_shoulder": 180, "right_shoulder": 20}
    else:
        ideal_angles = {"leg_raise": 90, "left_shoulder": 180, "right_shoulder": 20}

    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if 1 <= pose_state <= 2:
        state_index = pose_state - 1
        state_max_accuracies[state_index] = max(state_max_accuracies[state_index], accuracy)

    if pose_state == 1:
        display_message = "Stand straight."
        if current_angles["leg_raise"] > 170 and 0 < current_angles["left_shoulder"] < 40:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Lift your leg to the side."
        if 80 < current_angles["leg_raise"] < 100 and 170 < current_angles["left_shoulder"] < 190:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                overall_accuracy = sum(state_max_accuracies) / 2
                display_message = f"Lord of Dance Pose Completed! Overall Accuracy: {overall_accuracy:.2f}%"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

# -----------------------
# Session Dispatcher
# -----------------------
def session_pose_sequence(landmarks, output_image):
    global current_pose, pose_state, session_index, session_poses, session_results, current_pose_accuracies, display_message, continue_session, state_max_accuracies

    if session_index >= len(session_poses):
        overall_session_accuracy = sum(session_results.values()) / len(session_results) if session_results else 0
        medal = ""
        if 70 <= overall_session_accuracy < 80:
            medal = "Bronze"
        elif 80 <= overall_session_accuracy < 85:
            medal = "Silver"
        elif 85 <= overall_session_accuracy < 92:
            medal = "Gold"
        elif overall_session_accuracy >= 92:
            medal = "Diamond"
        display_message = f"Session Completed! Overall Accuracy: {overall_session_accuracy:.2f}%. Award: {medal}"
        continue_session = False
        return output_image

    if current_pose == "Tree Pose":
        output_image = tree_pose_sequence(landmarks, output_image)
    elif current_pose == "Warrior 1":
        output_image = warrior1_sequence(landmarks, output_image)
    elif current_pose == "Warrior 2":
        output_image = warrior2_sequence(landmarks, output_image)
    elif current_pose == "Triangle Pose":
        output_image = triangle_sequence(landmarks, output_image)
    elif current_pose == "Lord of Dance Pose":
        output_image = lord_of_dance_sequence(landmarks, output_image)

    if pose_state == 6:
        num_states = {"Tree Pose": 5, "Warrior 1": 3, "Warrior 2": 3, "Triangle Pose": 2, "Lord of Dance Pose": 2}
        pose_accuracy = sum(state_max_accuracies) / num_states[current_pose]
        session_results[current_pose] = pose_accuracy
        session_index += 1
        if session_index < len(session_poses):
            current_pose = session_poses[session_index]
            display_message = f"Starting {current_pose}..."
            pose_state = 1
            current_pose_accuracies = []
            state_max_accuracies = [0] * num_states[current_pose]
        else:
            overall_session_accuracy = sum(session_results.values()) / len(session_results) if session_results else 0
            medal = ""
            if 70 <= overall_session_accuracy < 80:
                medal = "Bronze"
            elif 80 <= overall_session_accuracy < 85:
                medal = "Silver"
            elif 85 <= overall_session_accuracy < 92:
                medal = "Gold"
            elif overall_session_accuracy >= 92:
                medal = "Diamond"
            display_message = f"Session Completed! Overall Accuracy: {overall_session_accuracy:.2f}%. Award: {medal}"
            continue_session = False

    return output_image

# -----------------------
# Webcam Feed with Pose Detection
# -----------------------
def webcam_feed():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1380)
    camera_video.set(4, 960)
    global continue_session, session_mode
    while camera_video.isOpened() and continue_session:
        ok, frame = camera_video.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        frame, landmarks = detectPose(frame, pose)
        if landmarks:
            if session_mode:
                frame = session_pose_sequence(landmarks, frame)
            else:
                if current_pose == "Tree Pose":
                    frame = tree_pose_sequence(landmarks, frame)
                elif current_pose == "Warrior 1":
                    frame = warrior1_sequence(landmarks, frame)
                elif current_pose == "Warrior 2":
                    frame = warrior2_sequence(landmarks, frame)
                elif current_pose == "Triangle Pose":
                    frame = triangle_sequence(landmarks, frame)
                elif current_pose == "Lord of Dance Pose":
                    frame = lord_of_dance_sequence(landmarks, frame)
                else:
                    frame = tree_pose_sequence(landmarks, frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera_video.release()
    cv2.destroyAllWindows()

# -----------------------
# Model Evaluation
# -----------------------
@app.route('/api/evaluate_model', methods=['GET', 'POST'])
def api_evaluate_model():
    if request.method == 'GET':
        return jsonify({
            'status': 'info',
            'message': 'This endpoint requires a POST request with JSON data containing landmarks_list and ideal_angles_list. Example: {"landmarks_list": [...], "ideal_angles_list": [...], "pose_type": "tree"}'
        }), 200

    try:
        if request.content_type != 'application/json':
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 415

        data = request.get_json()
        if not data or 'landmarks_list' not in data or 'ideal_angles_list' not in data:
            return jsonify({
                'error': 'Missing required parameters: landmarks_list and ideal_angles_list'
            }), 400

        landmarks_list = data['landmarks_list']
        ideal_angles_list = data['ideal_angles_list']
        pose_type = data.get('pose_type', 'tree')  # Default to tree pose

        # Map pose types to their sequence functions
        pose_sequences = {
            'tree': tree_pose_sequence,
            'warrior1': warrior1_sequence,
            'warrior2': warrior2_sequence,
            'triangle': triangle_sequence,
            'lord_of_dance': lord_of_dance_sequence
        }

        pose_sequence_func = pose_sequences.get(pose_type, tree_pose_sequence)

        # Validate input data
        if not isinstance(landmarks_list, list) or not isinstance(ideal_angles_list, list):
            return jsonify({
                'error': 'landmarks_list and ideal_angles_list must be lists'
            }), 400

        if len(landmarks_list) != len(ideal_angles_list):
            return jsonify({
                'error': 'landmarks_list and ideal_angles_list must have the same length'
            }), 400

        # Run evaluation
        metrics = evaluate_pose_model(landmarks_list, ideal_angles_list, pose_sequence_func)
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def evaluate_pose_model(landmarks_list, ideal_angles_list, pose_sequence_func, threshold=10):
    """
    Evaluate the pose detection model by calculating accuracy, precision, recall, and F1-score.
    
    Parameters:
    - landmarks_list: List of detected landmarks from multiple frames.
    - ideal_angles_list: List of ideal angles for each frame (ground truth).
    - pose_sequence_func: Function to process pose sequence (e.g., tree_pose_sequence).
    - threshold: Acceptable angle deviation (degrees) for a "correct" detection.
    
    Returns:
    - dict: Metrics including accuracy, precision, recall, F1-score, and avg processing time.
    """
    if not landmarks_list or not ideal_angles_list:
        raise ValueError("Empty input lists provided")

    tp, fp, fn = 0, 0, 0
    accuracies = []
    processing_times = []
    
    for landmarks, ideal_angles in zip(landmarks_list, ideal_angles_list):
        try:
            # Validate landmarks format
            if len(landmarks) < max(mp_pose.PoseLandmark) + 1:
                continue
                
            frame = np.zeros((960, 1280, 3), dtype=np.uint8)  # Dummy frame
            start_time = time.time()
            output_image = pose_sequence_func(landmarks, frame)
            end_time = time.time()
            
            # Extract current angles with error checking
            current_angles = {}
            try:
                current_angles = {
                    "left_knee": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
                    "right_knee": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]),
                    "left_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
                    "right_shoulder": calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
                }
            except IndexError:
                continue
                
            accuracy = calculate_accuracy(current_angles, ideal_angles)
            accuracies.append(accuracy)
            
            # Determine TP, FP, FN based on angle deviation
            for joint, ideal in ideal_angles.items():
                current = current_angles.get(joint, 0)
                deviation = abs(ideal - current)
                if deviation <= threshold:
                    if accuracy >= 70:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if accuracy >= 70:
                        fp += 1
            
            processing_times.append(end_time - start_time)
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue
    
    # Calculate metrics with safety checks
    total_predictions = tp + fp + fn
    accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
    
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "avg_processing_time": round(avg_processing_time, 4),
        "fps": round(fps, 2),
        "frames_processed": len(accuracies),
        "frames_total": len(landmarks_list)
    }

# -----------------------
# Flask Routes
# -----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yoga_try')
def yoga_try():
    global session_mode, continue_session, current_pose, pose_state, current_pose_accuracies, display_message, state_max_accuracies
    continue_session = True
    pose_state = 1
    current_pose_accuracies = []
    num_states = {"Tree Pose": 5, "Warrior 1": 3, "Warrior 2": 3, "Triangle Pose": 2, "Lord of Dance Pose": 2}
    state_max_accuracies = [0] * num_states.get(current_pose, 5)
    selected_pose = request.args.get('pose')
    if selected_pose:
        session_mode = False
        current_pose = selected_pose
        state_max_accuracies = [0] * num_states.get(current_pose, 5)
    pose_images = {
        "Tree Pose": "Tree pose.png",
        "Warrior 1": "Warrior 1.png",
        "Warrior 2": "Warrior 2.png",
        "Triangle Pose": "Triangle pose.png",
        "Lord of Dance Pose": "LordOfDance.png",
        "Trikonasana": "Trikonasana.jpg",
        "Virabadrasana": "Virabadrasana.png",
        "Vrikshasana": "Vrikshasana.png",
        "Bhujangasana": "Bhujangasana.png",
        "Sukhasana": "Sukhasana.png",
        "Chakrasana": "Chakrasana.png",
        "Balasana": "Balasana.png",
        "Shavasana": "Shavasana.png"
    }
    pose_img = pose_images.get(current_pose, "Tree pose.png")
    return render_template('yoga_try.html', pose_img=pose_img, current_pose=current_pose, display_message=display_message)

@app.route('/video_feed1')
def video_feed1():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_message')
def pose_message():
    return display_message

@app.route('/session')
def session():
    return render_template('session.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    global session_mode, session_poses, session_index, current_pose, session_results, pose_state, current_pose_accuracies, continue_session, state_max_accuracies
    selected = request.form.getlist('poses')
    if not selected:
        return redirect(url_for('session'))
    session_poses = selected
    session_mode = True
    session_index = 0
    current_pose = session_poses[0]
    session_results = {}
    pose_state = 1
    current_pose_accuracies = []
    num_states = {"Tree Pose": 5, "Warrior 1": 3, "Warrior 2": 3, "Triangle Pose": 2, "Lord of Dance Pose": 2}
    state_max_accuracies = [0] * num_states[current_pose]
    continue_session = True
    return redirect(url_for('yoga_try'))

@app.route('/session_results')
def session_results_page():
    overall_session_accuracy = sum(session_results.values()) / len(session_results) if session_results else 0
    medal = ""
    if 70 <= overall_session_accuracy < 80:
        medal = "Bronze"
    elif 80 <= overall_session_accuracy < 85:
        medal = "Silver"
    elif 85 <= overall_session_accuracy < 92:
        medal = "Gold"
    elif overall_session_accuracy >= 92:
        medal = "Diamond"
    
    # Save the session result
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'poses': list(session_results.keys()),
        'accuracies': session_results,
        'overall_accuracy': overall_session_accuracy,
        'medal': medal
    }
    session_history.append(session_data)
    
    return render_template('session_results.html', 
                         results=session_results, 
                         overall_accuracy=overall_session_accuracy, 
                         medal=medal)

@app.route('/meditation')
def meditation():
    return render_template('meditation.html')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/session_history', methods=['GET'])
def get_session_history():
    try:
        return jsonify(session_history)
    except Exception as e:
        print(f"Error getting session history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save_session_result', methods=['GET', 'POST', 'OPTIONS'])
def save_session_result():
    if request.method == 'OPTIONS':
        return '', 204
    
    if request.method == 'GET':
        return jsonify({
            'status': 'info',
            'message': 'This endpoint requires a POST request with JSON data containing session information. Example: {"poses": ["Tree Pose"], "accuracies": {"Tree Pose": 85.5}, "duration": 120}'
        }), 200
        
    try:
        if request.content_type != 'application/json':
            return jsonify({
                'status': 'error',
                'message': 'Content-Type must be application/json'
            }), 415
            
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
            
        print("Received session data:", data)  # Debug print
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'poses': data.get('poses', []),
            'accuracies': data.get('accuracies', {}),
            'duration': data.get('duration', 0),
            'medal': get_medal_for_accuracy(list(data.get('accuracies', {}).values())[0] if data.get('accuracies', {}) else 0)
        }
        session_history.append(session_data)
        print("Session saved:", session_data)  # Debug print
        return jsonify({'status': 'success', 'data': session_data})
    except Exception as e:
        print("Error saving session:", str(e))  # Debug print
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/meditation/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/meditation', filename)

@app.route('/api/meditation/sounds')
def get_meditation_sounds():
    sounds = [
        {
            'id': 'rain',
            'name': 'Gentle Rain',
            'source': '/meditation/audio/nature/rain.mp3',
            'icon': '🌧️'
        },
        {
            'id': 'forest',
            'name': 'Forest Ambiance',
            'source': '/meditation/audio/nature/forest.mp3',
            'icon': '🌳'
        },
        {
            'id': 'waves',
            'name': 'Ocean Waves',
            'source': '/meditation/audio/nature/waves.mp3',
            'icon': '🌊'
        },
        {
            'id': 'breeze',
            'name': 'Soft Breeze',
            'source': '/meditation/audio/nature/breeze.mp3',
            'icon': '🍃'
        },
        {
            'id': 'stream',
            'name': 'Flowing Stream',
            'source': '/meditation/audio/nature/stream.mp3',
            'icon': '💧'
        }
    ]
    return jsonify(sounds)

def get_medal_for_accuracy(accuracy):
    if accuracy >= 92:
        return "Diamond"
    elif accuracy >= 85:
        return "Gold"
    elif accuracy >= 80:
        return "Silver"
    elif accuracy >= 70:
        return "Bronze"
    return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)