# A.C.R.A.---S.E.
A.C.R.A. - S.E. is the part of the final year AI project, which was designed by Mykhailo Yuriev and El Chakib. The robotic arm features the bearingless interconnection between joints of the palm of the arm, as well as the finely tuned SVC model, which calculates the angles of each human finger through the camera, and maps them to move to a position




# ML Hand Gesture to Servo Control

This project uses **OpenCV**, **MediaPipe**, and a **Support Vector Machine (SVM)** model to track hand landmarks in real time, extract joint flexion angles, and map them to servo motor angles on a Raspberry Pi over UDP.

##  Features

* Real-time hand tracking using [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands).
* Angle extraction from finger joints.
* SVM classifiers trained to predict servo angles from flexion angles.
* UDP communication with Raspberry Pi for servo control.
* Smooth prediction updates with exponential filtering (`alpha` blending).

##  Project Structure

```
├── data.npy              # Training features (finger joint flexion angles)
├── labels.npy            # Training labels (servo angles)
├── svc_model.joblib      # Saved trained classifiers
├── hand_control.py       # Main script
```

##  Requirements

* Python 3.8+
* Libraries:

  ```bash
  pip install opencv-python mediapipe scikit-learn joblib numpy
  ```

##  Training the Model

If no saved model (`svc_model.joblib`) is found, the script will:

1. Load training data (`data.npy`, `labels.npy`).
2. Train **5 SVM classifiers** (one per finger/servo).
3. Save the trained models to `svc_model.joblib`.

Each model maps **hand joint angles → servo rotation angles (0–180°)**.

##  Running the Script

1. Connect Raspberry Pi to the same Wi-Fi network.
2. Update the IP and port in the script:

   ```python
   PI_IP = '172.27.234.132'  
   PI_PORT = 9999
   ```
3. Run the program:

   ```bash
   python hand_control.py
   ```
4. Press **Esc** to exit.

##  Communication Protocol

* Uses **UDP socket**.
* Sends 5 servo angles packed as **unsigned bytes** (`struct.pack('5B', *servo_vals)`).
* Raspberry Pi should listen for incoming packets and update servos accordingly.

##  Key Functions

* `calc_angle(a, b, c)`: Computes the angle at joint `b`.
* `extract_features(lm)`: Extracts 5 joint flexion angles (thumb, index, middle, ring, pinky).
* `gradient smoothing`: Exponential moving average (`alpha=0.3`) for stable servo movement.

##  Demo

1. Start webcam.
2. Move your hand — the program tracks landmarks.
3. Servo angles update on Raspberry Pi in real time.

---


## Hardware Setup

This project has **two parts**:

### 1. Laptop (Sender)

* Runs `hand_control.py`
* Uses **OpenCV + MediaPipe** to track hand gestures.
* Extracts joint angles → predicts servo angles (0–180°).
* Sends **5 servo angles** over **UDP** to the Raspberry Pi.

### 2. Raspberry Pi (Receiver + Servo Driver)

* Runs `pi_servo_receiver.py`
* Listens for UDP packets from the laptop.
* Controls **5 hobby servos** through a **PCA9685-based Servo HAT** (e.g., Waveshare or Adafruit).

###  Servo Wiring (PCA9685 HAT → Servos)

* **Channels 0–4** of the PCA9685 are used (one per finger).
* Each channel has 3 pins:

  * **V+ (red)** → Servo power (5V).
  * **GND (black/brown)** → Servo ground.
  * **PWM (yellow/orange)** → Control signal.
* Connect the servo leads directly to the corresponding channel slots:

  * Servo 1 → Channel 0
  * Servo 2 → Channel 1
  * Servo 3 → Channel 2
  * Servo 4 → Channel 3
  * Servo 5 → Channel 4

 **Power Note**:

* The PCA9685 HAT **must be powered externally** (typically 5–6V supply capable of handling total servo current).
* Servos should not be powered directly from the Raspberry Pi’s 5V pin.

