from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import mediapipe as mp
import math
import time

app = Flask(__name__)

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

# Session management variables
session_mode = False
session_poses = []         # List of poses for the session
session_index = 0          # Which pose in the session we are on
session_results = {}       # Pose name -> average accuracy
current_pose_accuracies = []  # Accumulate accuracy values for current pose
current_pose = "Tree Pose" # Default pose (will be set by query parameter)

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
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies
    ideal_angles = {
        "left_knee": 180,
        "right_knee": 180,
        "left_shoulder": 180,
        "right_shoulder": 180
    }
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
    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if pose_state == 1:
        display_message = "Stand still."
        if 165 < current_angles["left_knee"] < 195 and 165 < current_angles["right_knee"] < 195:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None

    elif pose_state == 2:
        display_message = "Lift your leg."
        if current_angles["left_knee"] < 60 or current_angles["right_knee"] < 60:
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
        if 15 < current_angles["left_shoulder"] < 40 and 15 < current_angles["right_shoulder"] < 40:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 5
                state_start_time = None
        else:
            state_start_time = None

    elif pose_state == 5:
        display_message = "Lower your leg to complete the pose."
        if 165 < current_angles["left_knee"] < 195 and 165 < current_angles["right_knee"] < 195:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                # In session mode, update final message later; for single-pose, we can show final message.
                display_message = "Tree Pose completed!"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def warrior1_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies
    ideal_angles = {"front_knee": 90, "back_leg": 180, "arms": 180}
    current_angles = {}
    current_angles["front_knee"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    current_angles["back_leg"] = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    current_angles["arms"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if pose_state == 1:
        display_message = "Stand with feet apart."
        if current_angles["back_leg"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Bend your front knee to 90°."
        if 80 < current_angles["front_knee"] < 100:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 3
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 3:
        display_message = "Raise your arms horizontally."
        if 170 < current_angles["arms"] < 190:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                display_message = "Warrior 1 Pose completed!"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def warrior2_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies
    ideal_angles = {"front_knee": 90, "back_leg": 180, "arms": 180}
    current_angles = {}
    current_angles["front_knee"] = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    current_angles["back_leg"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    current_angles["arms"] = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if pose_state == 1:
        display_message = "Stand with feet apart."
        if current_angles["back_leg"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Bend your front knee to 90°."
        if 80 < current_angles["front_knee"] < 100:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 3
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 3:
        display_message = "Extend your arms outward."
        if 170 < current_angles["arms"] < 190:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                display_message = "Warrior 2 Pose completed!"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def triangle_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies
    ideal_angles = {"side_stretch": 90, "leg": 180}
    current_angles = {}
    current_angles["side_stretch"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    current_angles["leg"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if pose_state == 1:
        display_message = "Stand with feet apart."
        if current_angles["leg"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Bend forward and reach toward your foot."
        if 80 < current_angles["side_stretch"] < 100:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                display_message = "Triangle Pose completed!"
                pose_state = 6
                state_start_time = None
        else:
            state_start_time = None

    cv2.putText(output_image, display_message, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    cv2.putText(output_image, accuracy_message, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    output_image = draw_hold_timer(output_image)
    return output_image

def lord_of_dance_sequence(landmarks, output_image):
    global pose_state, display_message, accuracy_message, state_start_time, current_pose_accuracies
    ideal_angles = {"leg_raise": 90, "arm_raise": 180}
    current_angles = {}
    current_angles["leg_raise"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    current_angles["arm_raise"] = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    accuracy = calculate_accuracy(current_angles, ideal_angles)
    accuracy_message = f"Accuracy: {accuracy:.2f}%"
    current_pose_accuracies.append(accuracy)

    if pose_state == 1:
        display_message = "Stand straight."
        if current_angles["arm_raise"] > 170:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                pose_state = 2
                state_start_time = None
        else:
            state_start_time = None
    elif pose_state == 2:
        display_message = "Lift your leg to the side."
        if 80 < current_angles["leg_raise"] < 100:
            if state_start_time is None:
                state_start_time = time.time()
            elif time.time() - state_start_time >= hold_duration:
                display_message = "Lord of Dance Pose completed!"
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
    global current_pose, pose_state, session_index, session_poses, session_results, current_pose_accuracies, display_message, continue_session

    # If all poses have been processed, update final message and stop
    if session_index >= len(session_poses):
        overall_accuracy = sum(session_results.values()) / len(session_results) if session_results else 0
        medal = ""
        if 80 <= overall_accuracy < 90:
            medal = "Bronze"
        elif 90 <= overall_accuracy < 93:
            medal = "Silver"
        elif 93 <= overall_accuracy < 96:
            medal = "Gold"
        elif overall_accuracy >= 96:
            medal = "Diamond"

        display_message = f"Session Completed! Overall Accuracy: {overall_accuracy:.2f}%. Award: {medal}"
        continue_session = False  # Stop the webcam feed
        return output_image

    # Process the current pose
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

    # When the current pose is completed, move to the next one
    if pose_state == 6:
        avg_accuracy = sum(current_pose_accuracies) / len(current_pose_accuracies) if current_pose_accuracies else 0
        session_results[current_pose] = avg_accuracy  
        session_index += 1  # Move to the next pose
        if session_index < len(session_poses):
            current_pose = session_poses[session_index]
            display_message = f"Starting {current_pose}..."
            pose_state = 1       # Reset state for the new pose
            current_pose_accuracies = []  # Reset accuracy tracking
        else:
            # Last pose has been completed – update final message.
            overall_accuracy = sum(session_results.values()) / len(session_results) if session_results else 0
            medal = ""
            if 80 <= overall_accuracy < 90:
                medal = "Bronze"
            elif 90 <= overall_accuracy < 93:
                medal = "Silver"
            elif 93 <= overall_accuracy < 96:
                medal = "Gold"
            elif overall_accuracy >= 96:
                medal = "Diamond"
            display_message = f"Session Completed! Overall Accuracy: {overall_accuracy:.2f}%. Award: {medal}"
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
# Flask Routes
# -----------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yoga_try')
def yoga_try():
    global session_mode, continue_session, current_pose, pose_state, current_pose_accuracies, display_message
    continue_session = True
    pose_state = 1
    current_pose_accuracies = []
    selected_pose = request.args.get('pose')
    if selected_pose:
        # Single-pose mode; disable session mode
        session_mode = False
        current_pose = selected_pose
    # Otherwise, leave session_mode as set by /start_session
    # Map each pose to its representative image file.
    pose_images = {
        "Tree Pose": "Tree pose.png",
        "Warrior 1": "Warrior1.jpg",
        "Warrior 2": "Warrior2.jpg",
        "Triangle Pose": "TrianglePose.jpg",
        "Lord of Dance Pose": "LordOfDance.jpg",
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
    global session_mode, session_poses, session_index, current_pose, session_results, pose_state, current_pose_accuracies, continue_session
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
    continue_session = True
    return redirect(url_for('yoga_try'))

@app.route('/session_results')
def session_results_page():
    overall_accuracy = sum(session_results.values()) / len(session_results) if session_results else 0
    medal = ""
    if 80 <= overall_accuracy < 90:
        medal = "Bronze"
    elif 90 <= overall_accuracy < 93:
        medal = "Silver"
    elif 93 <= overall_accuracy < 96:
        medal = "Gold"
    elif overall_accuracy >= 96:
        medal = "Diamond"
    return render_template('session_results.html', results=session_results, overall_accuracy=overall_accuracy, medal=medal)

@app.route('/meditation')
def meditation():
    return render_template('meditation.html')

if __name__ == '__main__':
    app.run(debug=True)
