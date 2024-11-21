import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Track both hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def track_hands(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Initialize dictionaries for hand landmarks
        right_hand_landmarks = None
        left_hand_landmarks = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Determine if left or right hand
                if results.multi_handedness[idx].classification[0].label == "Right":
                    right_hand_landmarks = hand_landmarks
                else:
                    left_hand_landmarks = hand_landmarks
                    
        return frame, right_hand_landmarks, left_hand_landmarks

    def get_gesture(self, landmarks):
        """
        Basic gesture recognition
        Returns: "thumbs_up", "thumbs_down", or None
        """
        if landmarks is None:
            return None
            
        # Convert landmarks to list of coordinates
        points = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        
        # Basic thumb up detection
        # Thumb tip is higher than thumb base for thumbs up
        if points[4][1] < points[2][1]:
            return "thumbs_up"
        # Thumb tip is lower than thumb base for thumbs down
        elif points[4][1] > points[2][1]:
            return "thumbs_down"
        
        return None

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Track hands
        frame, right_hand, left_hand = tracker.track_hands(frame)
        
        # Get gestures
        right_gesture = tracker.get_gesture(right_hand)
        left_gesture = tracker.get_gesture(left_hand)
        
        # Display gestures (if any)
        if right_gesture:
            cv2.putText(frame, f"Right: {right_gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if left_gesture:
            cv2.putText(frame, f"Left: {left_gesture}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()