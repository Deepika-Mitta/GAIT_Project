import cv2
import mediapipe as mp
import numpy as np
from joblib import load

# Load trained model
model = load('alphabet_model.joblib')

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

print("Press 'Esc' to exit the program.")

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
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract and normalize landmarks
            data_aux = []
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            box_width = max(x_) - min(x_)
            box_height = max(y_) - min(y_)
            for lm in hand_landmarks.landmark:
                data_aux.append((lm.x - min(x_)) / box_width)
                data_aux.append((lm.y - min(y_)) / box_height)

            # Bounding box coordinates
            x1 = max(int(min(x_) * W) - 10, 0)
            y1 = max(int(min(y_) * H) - 10, 0)
            x2 = min(int(max(x_) * W) + 10, W)
            y2 = min(int(max(y_) * H) + 10, H)

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]  # Directly use the predicted label

            # Draw bounding box and display the prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Sign Language Recognition', frame)

    # Exit on pressing 'Esc'
    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
