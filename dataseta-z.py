import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './alphabet_data'

# Data storage
data = []
labels = []

for letter in os.listdir(DATA_DIR):
    letter_dir = os.path.join(DATA_DIR, letter)
    if not os.path.isdir(letter_dir):
        continue

    for img_path in os.listdir(letter_dir):
        img_full_path = os.path.join(letter_dir, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                box_width = max(x_) - min(x_)
                box_height = max(y_) - min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.append((lm.x - min(x_)) / box_width)
                    data_aux.append((lm.y - min(y_)) / box_height)

                data.append(data_aux)
                labels.append(letter)

# Save the dataset
with open('alphabet_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
hands.close()
