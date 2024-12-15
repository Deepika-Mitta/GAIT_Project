import cv2
import mediapipe as mp
import numpy as np

# class ASLDetector:
#     def __init__(self):
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.7
#         )
#         self.mp_draw = mp.solutions.drawing_utils
#         self.landmarks = None  # Add this line
#         self.current_letter = None  # Add this line

#     def process_frame(self, frame):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb_frame)
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 self.landmarks = hand_landmarks  # Store landmarks
#                 self.mp_draw.draw_landmarks(
#                     frame, 
#                     hand_landmarks, 
#                     self.mp_hands.HAND_CONNECTIONS
#                 )
                
#                 # Draw finger states for debugging
#                 states = self.get_finger_states(hand_landmarks)
#                 for i, state in enumerate(states):
#                     cv2.putText(
#                         frame,
#                         f"F{i}: {state}",
#                         (10, 30 + i * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6,
#                         (255, 0, 0),
#                         1
#                     )
                
#                 self.current_letter = self.detect_letter(hand_landmarks)  # Store current letter
#                 if self.current_letter:
#                     cv2.putText(
#                         frame,
#                         f"Letter: {self.current_letter}",
#                         (10, 200),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         2,
#                         (0, 255, 0),
#                         2
#                     )
#         else:
#             self.landmarks = None  # Reset when no hands detected
#             self.current_letter = None
        
#         return frame

#     # Rest of the methods remain the same
#     def get_finger_states(self, landmarks):
#         """Returns the state of each finger (extended or not)"""
#         points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
#         states = []
        
#         thumb_extended = points[4][0] < points[3][0]
#         states.append(thumb_extended)
        
#         for finger_base in [5, 9, 13, 17]:
#             finger_tip = finger_base + 3
#             finger_extended = points[finger_tip][1] < points[finger_base][1]
#             states.append(finger_extended)
            
#         return states

#     def detect_letter(self, landmarks):
#         if landmarks is None:
#             return None
            
#         states = self.get_finger_states(landmarks)
#         points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
#         if not any(states[1:]) and not states[0]:
#             return 'A'
            
#         if all(states[1:]) and not states[0]:
#             return 'B'
            
#         if states[1] and not any(states[2:]) and not states[0]:
#             return 'D'
            
#         if all(states):
#             return 'E'
            
#         distance_index_thumb = np.linalg.norm(points[4] - points[8])
#         if distance_index_thumb < 0.05 and not any(states[2:]):
#             return 'F'
            
#         return None
    
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
        
        # Calibration values
        self.calibration_mode = False
        self.finger_base_positions = {}
        
        # Stabilization
        self.letter_history = []
        self.history_size = 10
        self.min_detection_confidence = 0.8  # 80% confidence required

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

    def get_finger_distances(self, landmarks):
        """Calculate distances between fingertips"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        distances = {}
        
        # Tips are points 4,8,12,16,20
        tips = [4, 8, 12, 16, 20]
        
        for i, tip1 in enumerate(tips):
            for tip2 in tips[i+1:]:
                distances[f"{tip1}-{tip2}"] = np.linalg.norm(points[tip1] - points[tip2])
                
        return distances
    def detect_letter(self, landmarks):
        if landmarks is None:
            return None
            
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        states = self.get_finger_states(landmarks)
        
        try:
            # A: Closed fist, thumb to side
            if (not any(states[1:]) and not states[0] and 
                self.get_finger_angle(points, 1, 2, 4) > 30):
                return 'A'
            
            # B: All fingers up, thumb tucked
            if (all(states[1:]) and not states[0] and
                all(self.is_finger_extended(points, base) for base in [5,9,13,17])):
                return 'B'
            
            # C: Curved fingers, thumb parallel
            thumb_idx_dist = np.linalg.norm(points[4] - points[8])
            if (0.1 < thumb_idx_dist < 0.2 and
                all(120 < self.get_finger_angle(points, base, base+1, base+2) < 150 
                    for base in [5,9,13,17])):
                return 'C'
            
            # D: Index up, others closed
            if (states[1] and not any(states[2:]) and 
                self.is_finger_extended(points, 5) and
                points[8][1] < points[5][1]):  # Index pointing up
                return 'D'
            
            # E: All fingers curled
            if (not any(states) and
                all(self.get_finger_angle(points, base, base+1, base+2) < 90 
                    for base in [5,9,13,17])):
                return 'E'
            
            # F: Index and thumb touching, others extended
            if (np.linalg.norm(points[4] - points[8]) < 0.05 and
                states[2] and states[3] and states[4]):
                return 'F'
        
            # G: Index pointing to side, thumb extended
            if states[1] and states[0] and not any(states[2:]):
                horizontal_diff = abs(points[8][0] - points[5][0])
                if horizontal_diff > 0.2:
                    return 'G'
            
            # H: Index and middle fingers straight, horizontal
            # if states[1] and states[2] and not states[3] and not states[4]:
            #     horizontal_diff = abs(points[8][0] - points[12][0])
            #     if horizontal_diff > 0.1:
            #         return 'H'
            
            # I: Pinky up, others closed
            if states[4] and not any(states[:4]):
                return 'I'
            
            # J: Moving motion of I (would need motion tracking)
            # K: Index and middle up in V shape, thumb at middle finger
            # if states[1] and states[2] and not any(states[3:]):
            #     finger_spread = np.linalg.norm(points[8] - points[12])
            #     if finger_spread > 0.1:
            #         return 'K'
            
            # L: L-shape with index and thumb
            if states[1] and states[0] and not any(states[2:]):
                if points[4][1] < points[3][1]:  # Thumb up
                    return 'L'
            
            # M: Three fingers over thumb
            if not states[0] and states[1] and states[2] and states[3] and not states[4]:
                if all(points[i][1] > points[i-2][1] for i in [8,12,16]):
                    return 'M'
            
            # N: Two fingers over thumb
            if not states[0] and states[1] and states[2] and not states[3] and not states[4]:
                if all(points[i][1] > points[i-2][1] for i in [8,12]):
                    return 'N'
            
            # O: Rounded O shape
            if all(np.linalg.norm(points[i] - points[i+3]) < 0.1 for i in [5,9,13,17]):
                return 'O'
            
            # P: Index pointing down, thumb out
            # if states[1] and not any(states[2:]) and states[0]:
            #     if points[8][1] > points[5][1]:  # Index pointing down
            #         return 'P'
            
            # Q: Index down diagonal
            # if states[1] and not any(states[2:]):
            #     if points[8][0] < points[5][0] and points[8][1] > points[5][1]:
            #         return 'Q'
            
            # R: Crossed fingers
            if states[1] and states[2] and not any(states[3:]):
                if np.linalg.norm(points[8] - points[12]) < 0.05:
                    return 'R'
            
            # S: Fist with thumb in front
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
            
            # T: Index hidden behind thumb
            if states[0] and not any(states[1:]):
                if points[4][0] > points[8][0]:
                    return 'T'
            
            # U: Index and middle parallel up
            if states[1] and states[2] and not any(states[3:]):
                if abs(points[8][0] - points[12][0]) < 0.05:
                    return 'U'
            
            # V: Index and middle in V
            # if states[1] and states[2] and not any(states[3:]):
            #     if np.linalg.norm(points[8] - points[12]) > 0.1:
            #         return 'V'
            
            # W: Index, middle, and ring fingers up
            if states[1] and states[2] and states[3] and not states[4]:
                return 'W'
            
            # X: Index hook
            # if states[1] and not any(states[2:]):
            #     if points[8][1] > points[7][1]:  # Tip below middle joint
            #         return 'X'
            
            # Y: Thumb and pinky out
            if states[0] and states[4] and not any(states[1:4]):
                return 'Y'
            
            # Z: Index drawing Z (would need motion tracking)
            
        except Exception as e:
            print(f"Error in letter detection: {e}")
            return None
            
        return None

    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand only
            self.landmarks = hand_landmarks
            
            # Draw hand landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2),
                self.mp_draw.DrawingSpec(color=(255,0,0), thickness=2)
            )
            
            # Detect letter
            detected_letter = self.detect_letter(hand_landmarks)
            
            # Update letter history for stabilization
            if detected_letter:
                self.letter_history.append(detected_letter)
                if len(self.letter_history) > self.history_size:
                    self.letter_history.pop(0)
                    
                # Check if letter is stable
                letter_counts = {}
                for letter in self.letter_history:
                    letter_counts[letter] = letter_counts.get(letter, 0) + 1
                    
                for letter, count in letter_counts.items():
                    confidence = count / len(self.letter_history)
                    if confidence >= self.min_detection_confidence:
                        self.current_letter = letter
                        
            # Display debug info
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            states = self.get_finger_states(hand_landmarks)
            
            # Display finger states
            for i, state in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
                cv2.putText(frame, f"{state}: {'Up' if states[i] else 'Down'}", 
                           (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255,0,0), 1)
            
            # Display detected letter
            if self.current_letter:
                cv2.putText(frame, f"Letter: {self.current_letter}", 
                           (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (0,255,0), 2)
            
            # Display confidence
            if self.letter_history:
                confidence = self.letter_history.count(self.current_letter) / len(self.letter_history)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0,255,0), 2)
        
        else:
            self.landmarks = None
            self.current_letter = None
            self.letter_history.clear()
            
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