import cv2
import mediapipe as mp
import numpy as np


class ASLDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on a single hand for better accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.landmarks = None
        self.current_letter = None

        # Stabilization
        self.letter_history = []
        self.history_size = 10
        self.min_detection_confidence = 0.8  # Confidence threshold for letter detection

    def get_finger_states(self, landmarks):
        """Get extended/closed state of each finger"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        states = []

        # Thumb
        thumb_angle = self.get_finger_angle(points, 2, 3, 4)
        thumb_extended = thumb_angle > 150  # Straighter angle indicates extension
        states.append(thumb_extended)

        # Other fingers
        for finger_base in [5, 9, 13, 17]:  # Index, middle, ring, pinky
            states.append(self.is_finger_extended(points, finger_base))

        return states

    def get_finger_angle(self, points, p1, p2, p3):
        """Calculate angle between three points"""
        v1 = points[p1] - points[p2]
        v2 = points[p3] - points[p2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def is_finger_extended(self, points, base_idx):
        """Check if a finger is extended using angles"""
        angle1 = self.get_finger_angle(points, base_idx, base_idx + 1, base_idx + 2)
        angle2 = self.get_finger_angle(points, base_idx + 1, base_idx + 2, base_idx + 3)
        return angle1 > 160 and angle2 > 160

    def detect_letter(self, landmarks):
        if landmarks is None:
            return None

        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        states = self.get_finger_states(landmarks)

        try:
            # S: Fist with thumb wrapped outside
            if not any(states[1:]) and states[0] and points[4][0] < points[3][0]:
                return 'S'

            # T: Thumb crosses over index
            if not any(states[1:]) and states[0] and points[4][0] > points[8][0]:
                return 'T'

            # P: Index pointing down, thumb out
            if states[1] and not any(states[2:]) and states[0] and points[8][1] > points[5][1]:
                return 'P'

            # H: Index and middle fingers straight, horizontal
            if states[1] and states[2] and not states[3] and not states[4]:
                horizontal_diff = abs(points[8][1] - points[12][1])
                if horizontal_diff < 0.05:
                    return 'H'

            # Q: Index pointing diagonally down-left
            if states[1] and not any(states[2:]) and points[8][0] < points[5][0] and points[8][1] > points[5][1]:
                return 'Q'

            # K: Index and middle in V shape, thumb at middle finger
            if states[1] and states[2] and not any(states[3:]):
                thumb_to_middle = np.linalg.norm(points[4] - points[12])
                if thumb_to_middle < 0.05:
                    return 'K'

            # V: Index and middle in wider V shape
            if states[1] and states[2] and not any(states[3:]):
                finger_spread = np.linalg.norm(points[8] - points[12])
                if finger_spread > 0.1:
                    return 'V'

            # X: Index hook
            if states[1] and not any(states[2:]) and points[8][1] > points[7][1]:
                return 'X'

        except Exception as e:
            print(f"Error in letter detection: {e}")
            return None

        return None



class ASLDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Changed to 1 for better accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.landmarks = None
        self.current_letter = None
        
        # Logging file
        self.log_file = "accuracy_log.csv"
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Ground Truth", "Detected Letter", "Correct"])

    def log_result(self, ground_truth, detected_letter):
        is_correct = "Yes" if ground_truth == detected_letter else "No"
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), ground_truth, detected_letter, is_correct])

    def process_frame(self, frame, ground_truth):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.landmarks = hand_landmarks
            
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                self.mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )
            
            detected_letter = self.detect_letter(hand_landmarks)  # Your letter detection method
            
            # Log the result
            self.log_result(ground_truth, detected_letter)
            
            # Display the result on the frame
            cv2.putText(frame, f"Detected: {detected_letter}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ground Truth: {ground_truth}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if ground_truth == detected_letter:
                cv2.putText(frame, "Correct", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Incorrect", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

# Update the main loop
def main():
    detector = ASLDetector()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit.")
    
    while cap.isOpened():
        ground_truth = input("Enter the ground truth letter (A-Z): ").upper()
        success, frame = cap.read()
        if not success:
            continue
        
        frame = detector.process_frame(frame, ground_truth)
        cv2.imshow('ASL Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

