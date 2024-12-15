import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from collections import deque
import statistics

# Load trained model
model = load('alphabet_model.joblib')

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

print("Press 'Esc' to exit the program.")

recognized_character = "None"  # To store the smoothed recognized character
prediction_window = deque(maxlen=10)  # Sliding window for predictions

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame not captured")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract and normalize landmarks
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            box_width = max(x_) - min(x_)
            box_height = max(y_) - min(y_)

            if box_width > 0 and box_height > 0:  # Ensure valid bounding box dimensions
                for lm in hand_landmarks.landmark:
                    data_aux.append((lm.x - min(x_)) / box_width)
                    data_aux.append((lm.y - min(y_)) / box_height)

                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0].upper()

                # Add prediction to the sliding window
                prediction_window.append(predicted_character)

                # Stabilize prediction using the most frequent value in the window
                recognized_character = statistics.mode(prediction_window)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Add a header at the top of the frame
    header_text = f"Character Recognized: {recognized_character if recognized_character else 'None'}"
    cv2.rectangle(frame, (0, 0), (W, 50), (0, 0, 0), -1)  # Black background for header
    cv2.putText(frame, header_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
