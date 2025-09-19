import cv2
import mediapipe as mp
import numpy as np
import socket
import struct
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os

PI_IP = '172.27.234.132'  # Raspberry Pi address on Wi-Fi LAN
PI_PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Angle calculation
def calc_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return abs(np.degrees(radians))

def extract_features(lm):
    finger_joint_indices = [
        (5, 6, 8),   # Index
        (9, 10, 12), # Middle
        (13, 14, 16),# Ring
        (17, 18, 20),# Pinky
        (1, 2, 4)    # Thumb
    ]
    feats = []
    for a, b, c in finger_joint_indices:
        ang = calc_angle((lm[a].x, lm[a].y), (lm[b].x, lm[b].y), (lm[c].x, lm[c].y))
        feats.append(ang)
    return np.array(feats)

MODEL_PATH = 'svc_model.joblib'
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)
else:
    data = np.load('data.npy')# raw flexion angles
    labels = np.load('labels.npy')# desired servo angles
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=666)
    
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel='rbf', probability=True))
    
    classifiers = [make_pipeline(StandardScaler(), SVC(kernel='rbf')) for _ in range(5)]
    for i in range(5):
        classifiers[i].fit(X_train, y_train[:, i])
    joblib.dump(classifiers, MODEL_PATH)
    clf = classifiers

cap = cv2.VideoCapture(0)

alpha = 0.3
prev_vals = np.zeros(5)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            feats = extract_features(lm)
            
            servo_vals = []
            for i, cls in enumerate(clf):
                pred = cls.predict([feats])[0]
               
                val = prev_vals[i] + alpha * (pred - prev_vals[i])
                prev_vals[i] = val
                servo_vals.append(int(np.clip(val, 0, 180)))
          
            data = struct.pack('5B', *servo_vals)
            sock.sendto(data, (PI_IP, PI_PORT))
            mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imshow('ML Hand Control', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    sock.close()