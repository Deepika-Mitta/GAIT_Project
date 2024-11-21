import cv2
import mediapipe as mp
import numpy as np

class ASLDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

    def get_finger_states(self, landmarks):
        """Returns the state of each finger (extended or not)"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Get states for each finger (thumb handled separately)
        states = []
        
        # Thumb (comparing tip to base)
        thumb_extended = points[4][0] < points[3][0]  # For right hand
        states.append(thumb_extended)
        
        # Other fingers
        for finger_base in [5, 9, 13, 17]:  # Index, middle, ring, pinky
            finger_tip = finger_base + 3
            finger_extended = points[finger_tip][1] < points[finger_base][1]
            states.append(finger_extended)
            
        return states

    def detect_letter(self, landmarks):
        if landmarks is None:
            return None
            
        states = self.get_finger_states(landmarks)
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Define letter patterns
        if not any(states[1:]) and not states[0]:  # All fingers closed
            return 'A'
            
        if all(states[1:]) and not states[0]:  # All fingers extended except thumb
            return 'B'
            
        if states[1] and not any(states[2:]) and not states[0]:  # Only index extended
            return 'D'
            
        if all(states):  # All fingers extended
            return 'E'
            
        # Add distance calculations for letters that need specific positioning
        distance_index_thumb = np.linalg.norm(points[4] - points[8])
        if distance_index_thumb < 0.05 and not any(states[2:]):  # Thumb and index touching
            return 'F'
            
        return None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Draw finger states for debugging
                states = self.get_finger_states(hand_landmarks)
                for i, state in enumerate(states):
                    cv2.putText(
                        frame,
                        f"F{i}: {state}",
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        1
                    )
                
                letter = self.detect_letter(hand_landmarks)
                if letter:
                    cv2.putText(
                        frame,
                        f"Letter: {letter}",
                        (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        2
                    )
        
        return frame

def main():
    detector = ASLDetector()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = detector.process_frame(frame)
        cv2.imshow('ASL Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()