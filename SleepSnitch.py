import cv2
import mediapipe as mp
import math
import time
import RPi.GPIO as GPIO  # For buzzer

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_EAR(eye_points):
    A = euclidean_distance(eye_points[1], eye_points[5])
    B = euclidean_distance(eye_points[2], eye_points[4])
    C = euclidean_distance(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Setup GPIO
BUZZER_PIN = 18  # Connect buzzer to GPIO18 (physical pin 12)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.21
CLOSED_FRAMES_THRESHOLD = 60
closed_frame_count = 0
buzzer_played = False

# Webcam start
cap = cv2.VideoCapture(0)  # USB webcam is usually 0 on Raspberry Pi
print("Press 'q' to quit")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        h, w = frame.shape[:2]

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark

            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            left_ear = calculate_EAR(left_eye)
            right_ear = calculate_EAR(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if avg_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0
                buzzer_played = False  # reset

            if closed_frame_count >= CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                if not buzzer_played:
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    time.sleep(0.5)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    buzzer_played = True

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

        cv2.imshow("Fatigue Detector - IOTIF Kit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
