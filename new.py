from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import threading
import time

# Initialize Flask app
app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Global variables for pose step tracking and message display
pose_state = 1
continue_session = True  # Determines if the user wants to continue

# Ideal angles for tree pose for each joint (example values)
IDEAL_ANGLES = {
    "left_knee": 180,
    "right_knee": 180,
    "left_shoulder": 180,
    "right_shoulder": 180
}

# Global variable to store the text message and accuracy to be displayed on the webpage
display_message = "Initializing pose tracking..."
accuracy_message = "Accuracy: 0%"

# Function to detect pose landmarks
def detectPose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
    
    return output_image, landmarks

# Function to calculate angle between three landmarks
def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Function to calculate accuracy based on ideal angles
def calculate_accuracy(current_angles):
    total_accuracy = 0
    for joint, ideal_angle in IDEAL_ANGLES.items():
        current_angle = current_angles.get(joint, 0)
        joint_accuracy = max(0, 100 - abs(ideal_angle - current_angle))
        total_accuracy += joint_accuracy
    return total_accuracy / len(IDEAL_ANGLES)

# Tree Pose Sequence Logic
def tree_pose_sequence(landmarks, output_image):
    global pose_state, continue_session, display_message, accuracy_message

    # Calculate angles for each joint
    current_angles = {}
    current_angles["left_knee"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    current_angles["right_knee"] = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    current_angles["left_shoulder"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    
    current_angles["right_shoulder"] = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Calculate accuracy
    accuracy = calculate_accuracy(current_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"

    # Pose validation and state progression
    if pose_state == 1:
        display_message = "Stand still."
        if 165 < current_angles["left_knee"] < 195 and 165 < current_angles["right_knee"] < 195:
            pose_state = 2

    elif pose_state == 2:
        display_message = "Lift your leg."
        if current_angles["left_knee"] < 60 or current_angles["right_knee"] < 60:
            pose_state = 3
    
    elif pose_state == 3:
        display_message = "Raise your hands and join them."
        if 170 < current_angles["left_shoulder"] < 190 and 170 < current_angles["right_shoulder"] < 190:
            pose_state = 4

    elif pose_state == 4:
        display_message = "Lower your hands."
        if 15 < current_angles["left_shoulder"] < 40 and 15 < current_angles["right_shoulder"] < 40:
            pose_state = 5

    elif pose_state == 5:
        display_message = "Lower your leg."
        if 165 < current_angles["left_knee"] < 195 and 165 < current_angles["right_knee"] < 195:
            display_message = "Tree Pose completed! Do you want to continue or stop?"
            time.sleep(2)
            continue_session = get_user_response()

            if continue_session:
                pose_state = 1
            else:
                pose_state = 6

    # Display the current step and accuracy on the frame
    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return output_image

# Placeholder for getting user response
def get_user_response():
    return True  # Here, it’s set to continue for testing purposes

# Webcam feed with pose detection
def webcam_feed():
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1380)
    camera_video.set(4, 960)

    while camera_video.isOpened() and continue_session:
        ok, frame = camera_video.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame, landmarks = detectPose(frame, pose)

        if landmarks:
            frame = tree_pose_sequence(landmarks, frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera_video.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yoga_try')
def yoga_try():
    return render_template('yoga_try.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_message')
def pose_message():
    return display_message

if __name__ == '__main__':
    app.run(debug=True)
