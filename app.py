
# Importing required packages and libraries
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import base64
import tempfile
import os

app = Flask(__name__, static_url_path='', static_folder='.')
app.config['SECRET_KEY'] = 'Nick'
socketio = SocketIO(app, async_mode='gevent')

# Initializing YOLO model for object detection
# Here 'best.pt' is our custum trained yolo model
model = YOLO('best.pt')

# Initializing mediapipe framework for movement tracking

# model to analyse face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# model to analyse pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Setting previous values that will help to calculate movement
previous_face_landmark = None
previous_pose_landmark = None

# Function to calculate face movement if present
def detect_face(image):
    global previous_face_landmark
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        current_face_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
        if previous_face_landmark is not None:
            movement = np.sqrt(np.sum(np.square(current_face_landmarks - previous_face_landmark)))
            previous_face_landmark = current_face_landmarks
            return movement
        else:
            previous_face_landmarks = current_face_landmarks
    return 0

# Function to calculate pose movement if present
def detect_pose(image):
    global previous_pose_landmark
    results = pose.process(image)
    if results.pose_landmarks:
        current_pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        if previous_pose_landmark is not None:
            movement = np.sqrt(np.sum(np.square(current_pose_landmarks - previous_pose_landmark)))
            previous_pose_landmarks = current_pose_landmarks
            return movement
        else:
            previous_pose_landmarks = current_pose_landmarks
    return 0

def detect_object(image):
    results = model(image)
    face_movement = 0
    pose_movement = 0

    for result in results:
        boxes = result.boxes
        if len(boxes) == 1 and boxes[0].cls[0] == 0:
            img = image.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_movement = detect_face(img)
            pose_movement = detect_pose(img)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, [face_movement, pose_movement]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['file']
    temp_video_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_video_path)
    process_video(temp_video_path)
    return 'Video uploaded successfully'

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_interval = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to reduce resolution
        frame = cv2.resize(frame, (640, 360))

        if count % frame_interval == 0:
            frame, movement = detect_object(frame)

            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Emit frame and movement data to frontend
            socketio.emit('frame', {'frame': frame_data, 'movement': movement, 'frame_number': count})

        count += 1

    cap.release()

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True, use_reloader=False)
