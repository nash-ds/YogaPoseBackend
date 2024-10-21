from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math
import pyttsx3
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize pyttsx3 engine for voice feedback
engine = pyttsx3.init()

# Set properties for pyttsx3
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change index for different voices
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

# Initialize pose state variable
pose_state = 1  # Start at step 1

# Lock for thread safety
audio_lock = threading.Lock()

# Function to provide voice instructions
def voice_assistant(message):
    print(f"Speaking: {message}")  # Debug statement
    def speak():
        with audio_lock:
            engine.say(message)
            engine.runAndWait()
    
    threading.Thread(target=speak).start()


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

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def classifyTreePose(landmarks, output_image):
    global pose_state
    
    # Calculate angles for Tree Pose checkpoints
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # Check pose step-by-step using the pose_state variable
    if pose_state == 1:
        if 165 < left_knee_angle < 195 and 165 < right_knee_angle < 195:
            voice_assistant("Please lift your leg.")
            pose_state = 2  # Move to next step
    
    elif pose_state == 2:
        if left_knee_angle < 60 or right_knee_angle < 60:
            voice_assistant("Raise your hands and join them.")
            pose_state = 3
    
    elif pose_state == 3:
        if 170 < left_shoulder_angle < 190 and 170 < right_shoulder_angle < 190:
            voice_assistant("Tree Pose completed, lower your hands.")
            pose_state = 4
    
    elif pose_state == 4:
        if 80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110:
            voice_assistant("Lower your leg.")
            pose_state = 5
    
    elif pose_state == 5:
        if 165 < left_knee_angle < 195 and 165 < right_knee_angle < 195:
            voice_assistant("Tree Pose completed!")
            pose_state = 1  # Reset for next cycle

    # Write the label of the current step on the output image.
    cv2.putText(output_image, f'Step {pose_state}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return output_image

# Function to read from the webcam and classify pose step-by-step
def webcam_feed():
    camera_video = cv2.VideoCapture('http://100.93.186.46:8080/video')
    camera_video.set(3, 1380)
    camera_video.set(4, 960)

    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame, landmarks = detectPose(frame, pose)

        if landmarks:
            frame = classifyTreePose(landmarks, frame)

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

if __name__ == '__main__':
    app.run(debug=True)
