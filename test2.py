import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from huggingface_hub import login

from joblib import load
from collections import deque
import statistics


class ASLAlphabetDetector:
    def __init__(self):
        self.model = load('alphabet_model.joblib')
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.prediction_window = deque(maxlen=10)
        self.current_letter = None

    def reset_detection_state(self):
        """Reset current letter and prediction window."""
        self.current_letter = None
        self.prediction_window.clear()
        # print("Detection state reset.")

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and normalize landmarks
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                box_width = max(x_) - min(x_)
                box_height = max(y_) - min(y_)

                if box_width > 0 and box_height > 0:
                    for lm in hand_landmarks.landmark:
                        data_aux.append((lm.x - min(x_)) / box_width)
                        data_aux.append((lm.y - min(y_)) / box_height)

                    # Make prediction
                    prediction = self.model.predict([np.asarray(data_aux)])
                    predicted_character = prediction[0].upper()

                    # Add prediction to the sliding window
                    self.prediction_window.append(predicted_character)

                    # Stabilize prediction using the most frequent value in the window
                    try:
                        self.current_letter = statistics.mode(self.prediction_window)
                    except statistics.StatisticsError:
                        # If no unique mode, use the most recent prediction
                        self.current_letter = predicted_character

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        else:
            self.current_letter = None

        return frame
    
def main():
    # Replace with your Hugging Face token
    # HUGGINGFACE_TOKEN = "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"
    
    try:
        predictor = ASLAlphabetDetector()
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            frame = predictor.process_frame(frame)
            cv2.imshow('ASL Word Prediction', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):  # Clear buffer
                predictor.letter_buffer = []
                predictor.suggestions = []
            elif key == ord(' '):  # Space to accept word
                if predictor.suggestions:
                    predictor.current_word = predictor.suggestions[0]
                    print(f"Accepted word: {predictor.current_word}")
                    predictor.letter_buffer = []
                    predictor.suggestions = []
                
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
