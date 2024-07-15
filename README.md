# Innovative-Monitoring-System---TeleICU

Intensive Care Units (ICUs) are critical environments where constant monitoring of patients is essential.
Traditional monitoring methods rely heavily on physical presence, which can be resource-intensive and prone to human error.
This project aims to introduce an automated, computer vision-based monitoring system to assist medical staff by providing continuous and accurate monitoring of patients' conditions.

# Objectives

•	To detect the presence of patients, doctors, nurses, and relatives in ICU rooms using YOLOv8.
•	To monitor the movement of patients when they are alone in the room.

# Methodology

## YOLOv8 for Individual Detection

•	Model Selection: YOLOv8, a state-of-the-art object detection model, was selected for its high accuracy and real-time performance capabilities.

•	Dataset Collection:
1.	Sources: Since the icu data is crucial for hospitals so we collected the data from online resources such as google images and istock. 
2.	Classes: The dataset has four classes: patient, doctor, nurse, and relatives.
3.	Annotation: For the annotation task we have used CVAT tool and export the annotation in yolo format.

•	Model Training:
1.	Training Configuration: The YOLOv8 model was trained using a high-performance GPU setup (Google Colab). Key parameters included a batch size of 16 and training for 300 epochs.

## MediaPipe for Movement and Expression Analysis

1.	Condition Check: If YOLOv8 detects that the patient is alone in the room then only movement of patient get tracked.
2.	Pose Tracking: MediaPipe's pose module is used to track the patient's body movements by calculating the difference between consecutive frames.
3.	Face Mesh Tracking: MediaPipe's face mesh module is used to track facial expressions and movements.

## Comparison of Training Approaches

To evaluate the effectiveness of using pre-trained weights versus training the model from scratch, two separate training approaches were employed:

### •	Training with Pre-trained Weights:

o	Initialization: The YOLOv8 model was initialized with pre-trained weights from the COCO dataset.
o	Fine-tuning: The model was fine-tuned on the custom dataset for detecting patients, doctors, nurses, and relatives.
o	Results: Fine-tuning the model with pre-trained weights led to faster convergence and higher initial accuracy, with the model reaching optimal performance after fewer epochs.

### •	Training from Scratch:

o	Initialization: The YOLOv8 model was initialized with random weights.
o	Full Training: The model was trained on the custom dataset from scratch.
o	Results: Training from scratch required significantly more epochs to reach comparable performance. However, it allowed the model to fully adapt to the specific characteristics of the custom dataset

## System Architecture

The system consists of a frontend interface, a backend server, and an API for communication between the two.
The frontend allows users to input a video file. The backend, implemented using Flask and ngrok for tunneling, handles video processing and serves the results to the frontend.

## Frontend

The frontend is designed to take user input in the form of a video file or a URL.
It displays the processed video and dynamically updates the graph showing the patient's movement and facial expression data.

## Backend

1.	YOLOv8 Integration: The backend loads a pre-trained YOLOv8 model and processes video frames to detect and classify individuals.
2.	MediaPipe Integration: When a patient is detected to be alone, the backend uses MediaPipe's pose and face mesh modules to calculate frame differences using euclidien distance and analyze movements and expressions.
3.	Data Handling and Plotting: The movement and expression data are sent to the frontend via an API, where it is plotted on a graph using a dynamic plotting library called chart.js.

## Dynamic Graph Plotting

The graph is updated in real-time to reflect the patient's movement and facial expression data.
This provides healthcare professionals with immediate insights into the patient's activity levels and potential issues.

# Results

The system was tested on video. 
YOLOv8 demonstrated moderate accuracy in detecting and classifying individuals.
MediaPipe effectively quantified patient body movements and facial expressions, and the dynamic graph plotting provided clear and immediate visualization of activity levels.
This monitoring system has the potential to enhance patient care by providing continuous and accurate monitoring.

# References

•	An automated ICU agitation monitoring system for video streaming using deep learning classification
•	YOLOv8: Ultralytics YOLOv8. Available at: https://github.com/ultralytics/ultralytics
•	MediaPipe: A Google framework for building multimodal applied machine learning pipelines. Available at: https://google.github.io/mediapipe/

