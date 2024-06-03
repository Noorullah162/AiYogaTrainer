from flask import Flask, Response, render_template, request, redirect, url_for,send_from_directory,send_file, session
import cv2
import os
import tensorflow as tf
import numpy as np
import enum
from typing import List, NamedTuple
import sys


pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)




class BodyPart(enum.Enum):
  """Enum representing human body keypoints detected by pose estimation models."""
  NOSE = 0
  LEFT_EYE = 1
  RIGHT_EYE = 2
  LEFT_EAR = 3
  RIGHT_EAR = 4
  LEFT_SHOULDER = 5
  RIGHT_SHOULDER = 6
  LEFT_ELBOW = 7
  RIGHT_ELBOW = 8
  LEFT_WRIST = 9
  RIGHT_WRIST = 10
  LEFT_HIP = 11
  RIGHT_HIP = 12
  LEFT_KNEE = 13
  RIGHT_KNEE = 14
  LEFT_ANKLE = 15
  RIGHT_ANKLE = 16
    
# print(BodyPart.LEFT_ANKLE)

edges = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

print(BodyPart.LEFT_ANKLE.value)
import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('movenet_thunder')

import os
import tensorflow as tf
import csv
import pandas as pd
from tensorflow import keras
import tensorflow as tf



def get_center_point(landmarks, left_bodypart, right_bodypart):
    """Calculates the center point of the two given landmarks."""
    left = tf.gather(landmarks, left_bodypart.value, axis=1)
    right = tf.gather(landmarks, right_bodypart.value, axis=1)
    center = left * 0.5 + right * 0.5
    return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
    """Calculates pose size.

    It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
    """
    # Hips center
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

    # Shoulders center
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

    # Torso size as the minimum body size
    torso_size = tf.linalg.norm(shoulders_center - hips_center)
    # Pose center
    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                     BodyPart.RIGHT_HIP)
    pose_center_new = tf.expand_dims(pose_center_new, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to
    # perform substraction
    pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

    # Dist to pose center
    d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
    # Max dist to pose center
    max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

    # Normalize scale
    pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size



def normalize_pose_landmarks(landmarks):
    """Normalizes the landmarks translation by moving the pose center to (0,0) and
    scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

    pose_center = tf.expand_dims(pose_center, axis=1)
    # Broadcast the pose center to the same size as the landmark vector to perform
    # substraction
    pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
    landmarks = landmarks - pose_center

    # Scale the landmarks to a constant pose size
    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size
    return landmarks


def landmarks_to_embedding(landmarks_and_scores):
    """Converts the input landmarks into a pose embedding."""
    # Reshape the flat input into a matrix with shape=(17, 3)
    reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

    # Normalize landmarks 2D
    landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
    # Flatten the normalized landmark coordinates into a vector
    embedding = keras.layers.Flatten()(landmarks)
    return embedding


def preprocess_data(data):
    embedding = landmarks_to_embedding(data)
    t = tf.reshape(embedding, (34))
    return t

from keras.models import load_model
import cv2
import pyttsx3
import time
import threading
import math
from queue import Queue
import pygame
from collections import Counter

pygame.mixer.init()

moving_hands_close = 50
moving_leg_close = 50
moving_handleg_close = 30
moving_handup_up = 400
bendhandleg = 300
hipleg = 80
model = load_model('model.h5')

handsup = pygame.mixer.Sound("handsup.mp3")
raiseleg = pygame.mixer.Sound("raiseleg.mp3")
keepdoing = pygame.mixer.Sound("correct.mp3")
handlegup = pygame.mixer.Sound("handleg.mp3")    
touchfoot = pygame.mixer.Sound("touchfoot.mp3")
chairpos = pygame.mixer.Sound("chair.mp3")
bend = pygame.mixer.Sound("bend.mp3")

class_names = ['Chair Pose', 'cobra', 'DownDog Pose', 'no_pose', 'Shoulder Pose', 'Triangle Pose', 'Tree Pose', 'Warrior Pose']
def webcamlive(posename):
    print(posename)
    cap=cv2.VideoCapture(0)
    # Collect threads for asynchronous feedback
    count = 1
    AnalysisFeedbackList = []
    while True:
        success, img = cap.read()  # Read frame from the camera
        if not success:
            break
        else:
            image = img.copy()
            person = movenet.detect(image,reset_crop_region=False)
            pose_landmarks = np.array( [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]  for keypoint in person.keypoints], dtype=np.float32)
            data = pose_landmarks.flatten().astype(float)
            bb_startpoints = (person.bounding_box[0].x, person.bounding_box[0].y)
            bb_endpoints = (person.bounding_box[1].x, person.bounding_box[1].y)
            color = (255, 0, 0)
            thickness = 3
            img = cv2.rectangle(img, bb_startpoints, bb_endpoints, color, thickness)
            data = np.array(data, dtype=np.float32)
            data = np.reshape(data, (1, 51))
            processed_data = preprocess_data(data)
            reshaped_data = tf.reshape(processed_data, ( -1, 34))
            confidence_threshold = 0.4
            for edge, colo in edges.items():
                p1, p2 = edge
                x1, y1, c1 = pose_landmarks[p1]
                x2, y2, c2 = pose_landmarks[p2]
                if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

            for kp in pose_landmarks:
                kx, ky, kp_conf = kp
                if kp_conf > confidence_threshold:
                    cv2.circle(img, (int(kx), int(ky)), 4, (0,255,0), -1)
            
            prediction = model.predict(reshaped_data)
            predicted_label = np.argmax(prediction)
            print("predicting label", predicted_label)
            if (count <= 20):
                AnalysisFeedbackList.append(predicted_label)
                count += 1
            else:
                occurence_count = Counter(AnalysisFeedbackList)
                Analysislabel = occurence_count.most_common(1)[0][0]
                # Get predicted label
                keypoint = person.keypoints
                # Check if predicted label is correct
                correct_label = class_names.index(posename)  # Replace with the correct label for the image
                print("**************************************************\n\n")
                print("Analysislabel = ",Analysislabel, end = "\n\n")
                print("**************************************************\n\n")
                if posename == "Tree Pose":
                    if Analysislabel != correct_label:
                        x1, y1 = keypoint[BodyPart.LEFT_WRIST.value].coordinate.x, keypoint[BodyPart.LEFT_WRIST.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_WRIST.value].coordinate.x, keypoint[BodyPart.RIGHT_WRIST.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        handDis = distance
                        x1, y1 = keypoint[BodyPart.LEFT_KNEE.value].coordinate.x, keypoint[BodyPart.LEFT_KNEE.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_ANKLE.value].coordinate.x, keypoint[BodyPart.RIGHT_ANKLE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        legDis = distance
                        print(handDis, legDis)
                        msg = ''
                        if((handDis > moving_hands_close) and (legDis > moving_leg_close)):
                            handlegup.play()
                            # msg = "move hands close and raise up leg"
                        elif (handDis > moving_hands_close):
                            handsup.play()
                            # msg = "move hands close"
                        elif (legDis > moving_leg_close):
                            raiseleg.play()
                            #msg = "raise up leg"
                    else:
                        keepdoing.play()
                        # msg = "Keep doing"
                elif posename == "Triangle Pose":
                    if Analysislabel != correct_label:
                        x1, y1 = keypoint[BodyPart.LEFT_WRIST.value].coordinate.x, keypoint[BodyPart.LEFT_WRIST.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.LEFT_ANKLE.value].coordinate.x, keypoint[BodyPart.LEFT_ANKLE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        handlegDis = distance
                        x1, y1 = keypoint[BodyPart.RIGHT_WRIST.value].coordinate.x, keypoint[BodyPart.RIGHT_WRIST.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_HIP.value].coordinate.x, keypoint[BodyPart.RIGHT_HIP.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        handDis = distance
                        print(handlegDis, handDis)
                        msg = ''
                        if((handlegDis > moving_handleg_close)):
                            touchfoot.play()
                            # msg = "touch your foot"
                        elif (handDis > moving_handup_up):
                            handsup.play()
                            # msg = "move hands up"
                        else:
                            keepdoing.play()
                    else:
                        keepdoing.play()
                elif posename == "DownDog Pose":
                    if Analysislabel != correct_label:
                        x1, y1 = keypoint[BodyPart.RIGHT_SHOULDER.value].coordinate.x, keypoint[BodyPart.RIGHT_SHOULDER.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_KNEE.value].coordinate.x, keypoint[BodyPart.RIGHT_KNEE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        handlegDis = distance
                        x1, y1 = keypoint[BodyPart.RIGHT_WRIST.value].coordinate.x, keypoint[BodyPart.RIGHT_WRIST.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_HIP.value].coordinate.x, keypoint[BodyPart.RIGHT_HIP.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        handDis = distance
                        msg = ''
                        if((handlegDis > bendhandleg)):
                            bend.play()
                            # msg = "touch your foot"
                    else:
                        keepdoing.play()
                elif posename == "Chair Pose":
                    if Analysislabel != correct_label:
                        x1, y1 = keypoint[BodyPart.RIGHT_HIP.value].coordinate.x, keypoint[BodyPart.RIGHT_HIP.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_ANKLE.value].coordinate.x, keypoint[BodyPart.RIGHT_ANKLE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        righthiplegDis = distance
                        x1, y1 = keypoint[BodyPart.LEFT_HIP.value].coordinate.x, keypoint[BodyPart.LEFT_HIP.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.LEFT_ANKLE.value].coordinate.x, keypoint[BodyPart.LEFT_ANKLE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        lefthiplegDis = distance
                        
                        x1, y1 = keypoint[BodyPart.RIGHT_HIP.value].coordinate.x, keypoint[BodyPart.RIGHT_HIP.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.RIGHT_WRIST.value].coordinate.x, keypoint[BodyPart.RIGHT_WRIST.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        righthandup = distance
                        
                        x1, y1 = keypoint[BodyPart.LEFT_HIP.value].coordinate.x, keypoint[BodyPart.LEFT_HIP.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.LEFT_WRIST.value].coordinate.x, keypoint[BodyPart.LEFT_WRIST.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        lefthandup = distance
                
                        msg = ''
                        if(righthiplegDis > hipleg ) or (lefthiplegDis > hipleg):
                            chairpos.play()
                            # msg = "Setting position"
                        elif(lefthandup > 150) or (righthandup > 150):
                            handsup.play()
                    else:
                        keepdoing.play()  
                elif posename == "Warrior Pose":
                    if Analysislabel != correct_label:
                        x1, y1 = keypoint[BodyPart.RIGHT_ANKLE.value].coordinate.x, keypoint[BodyPart.RIGHT_ANKLE.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.LEFT_ANKLE.value].coordinate.x, keypoint[BodyPart.LEFT_ANKLE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        legdis = distance
                        x1, y1 = keypoint[BodyPart.RIGHT_WRIST.value].coordinate.x, keypoint[BodyPart.RIGHT_WRIST.value].coordinate.y
                        x2, y2 = keypoint[BodyPart.LEFT_ANKLE.value].coordinate.x, keypoint[BodyPart.LEFT_ANKLE.value].coordinate.y
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        handup = distance
                        
                        msg = ''
                        if(legdis > 180 ):
                            raiseleg.play()
                            # msg = "Setting position"
                        elif(handup > 160):
                            handsup.play()
                    else:
                        keepdoing.play()  
                count = 1
                AnalysisFeedbackList.clear()
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n') 


import time
from datetime import datetime
app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
# app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/tutorials')
def tutorials():
    tutorialspoints = [
        '1. When App ask for permission of camera, allow it to access to capture pose.',
        '2. Select what pose you want to do in the dropdown.',
        '3. Read Instructions of that pose so you will know how to do that pose.',
        '4. Click on Start pose and see the image of the that pose in the right side and replicate that image in front of camera.',
        '5. If you will do correctly the skeleton over the video will become green in color and sound will start playing'
    ]

    fix_camera_points = [
        'Solution 1. Make sure you have allowed the permission of camera, if you have denied the permission, go to setting of your browser to allow the access of camera to the application.',
        'Solution 2. Make sure no any other application is not accessing camera at that time, if yes, close that application',
        'Solution 3. Try to close all the other opened browsers'
    ]
    return render_template('tutorials.html', tutorialspoints=tutorialspoints, fix_camera_points=fix_camera_points)

@app.route('/pose')
def pose():
    # code to start the tutorial goes here
    return render_template('pose.html')

@app.route('/selected_pose', methods=['POST'])
def selected_pose():
    selected_pose = request.form['selected_pose']
#     print(selected_pose)
    # Redirect to yoga.html with the selected pose as a URL parameter
    return redirect(url_for('yoga', pose=selected_pose))

@app.route('/yoga')
def yoga():
    # code to start the tutorial goes here
    selected_pose = request.args.get('pose')
    pose = selected_pose.split()
    print(pose[0])
    start = datetime.now()
    return render_template('yoga.html', pose=selected_pose, posename = pose[0], starttime = start)

@app.route('/video_feed/<string:posename>')
def video_feed(posename):
    print("helloworld", posename)
    return Response(webcamlive(posename), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run()
    