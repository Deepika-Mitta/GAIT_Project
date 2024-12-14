import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from huggingface_hub import login
from test2 import ASLDetector



class LlamaPredictor:
    def __init__(self, token):
        login(token)
        try:
            print("Loading tokenizer and model...")
            # Using a smaller model for faster responses
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=token
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_suggestions(self, letters):
        if not letters:
            return []
            
        try:
            prefix = ''.join(letters)
            print(f"Generating suggestions for prefix: {prefix}")
            
            prompt = f"""<human>Generate 3 common English words that start with '{prefix}'. Reply with just the words separated by commas.</human>
            <assistant>"""
            
            # Set a timeout for generation
            with torch.no_grad():
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = inputs.to(self.model.device)
                
                print("Generating response...")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,  # Shorter output
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_k=10,  # More focused sampling
                    num_beams=1,  # No beam search for speed
                    early_stopping=True
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Raw response: {response}")
                
                # Clean up response
                response = response.split('<assistant>')[-1].strip()
                words = [word.strip() for word in response.split(',')]
                valid_words = [w for w in words if w.lower().startswith(prefix.lower())]
                print(f"Processed suggestions: {valid_words[:3]}")
                return valid_words[:3]
                
        except Exception as e:
            print(f"Error in suggestion generation: {e}")
            return []
        

class ASLWordPredictor:
    def __init__(self, token):
        try:
            # Basic initialization
            self.detector = ASLDetector()
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            
            # State variables
            self.STATE_LETTER_DETECTION = "letter_detection"
            self.STATE_GESTURE_DETECTION = "gesture_detection"
            self.current_state = self.STATE_LETTER_DETECTION
            
            # Word tracking
            self.letter_buffer = []
            self.current_word = ""
            self.suggestions = []
            self.accepted_suggestions = []
            self.rejected_suggestions = set()
            self.current_suggestion_index = 0
            
            # Timing and stability variables
            self.last_detection_time = time.time()
            self.last_gesture_time = 0
            self.detection_cooldown = 1.5
            self.stable_pose_duration = 1.0
            self.gesture_cooldown = 2.0
            self.cooldown = 1.0  # Added missing cooldown attribute
            self.last_stable_pose = None
            self.stable_pose_start_time = None
            self.possible_words = []  # Add this to store accepted word list
            self.waiting_for_next_letter = False  # Add this to track if we're waiting for next letter

            
            print("Initializing Llama model...")
            self.llama = LlamaPredictor(token)
            print("Model initialization complete")
            
        except Exception as e:
            print(f"Error in initialization: {e}")
            raise

    def get_new_suggestions(self, prefix):
        """Get suggestions excluding previously rejected ones"""
        try:
            all_suggestions = self.llama.get_suggestions(prefix)
            return [s for s in all_suggestions if s not in self.rejected_suggestions]
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []

    
    def process_frame(self, frame):
        try:
            current_time = time.time()
            
            if self.current_state == self.STATE_LETTER_DETECTION:
                frame = self.detector.process_frame(frame)
                
                if self.detector.current_letter:
                    if self.stable_pose_start_time is not None:
                        stability_progress = min(1.0, (current_time - self.stable_pose_start_time) / self.stable_pose_duration)
                        progress_width = int(200 * stability_progress)
                        cv2.rectangle(frame, (10, 270), (210, 290), (0, 0, 255), 2)
                        cv2.rectangle(frame, (10, 270), (10 + progress_width, 290), (0, 255, 0), -1)
                    
                    if self.is_pose_stable(self.detector.current_letter):
                        if current_time - self.last_detection_time > self.detection_cooldown:
                            new_letter = self.detector.current_letter
                            
                            if self.waiting_for_next_letter and self.possible_words:
                                current_pos = len(self.letter_buffer)
                                matching_words = [word for word in self.possible_words 
                                               if len(word) > current_pos and 
                                               word[current_pos].upper() == new_letter]
                                
                                if matching_words:
                                    self.letter_buffer.append(new_letter)
                                    self.possible_words = matching_words
                                    
                                    # Check if we're down to one word
                                    if len(matching_words) == 1:
                                        self.current_word = matching_words[0]
                                        print(f"Word locked: {self.current_word}")
                                        self.reset_state()
                                        return frame
                                    else:
                                        print(f"Letter accepted: {new_letter}")
                                        print(f"Remaining possible words: {self.possible_words}")
                                        self.waiting_for_next_letter = True
                                
                            elif not self.waiting_for_next_letter:
                                if not self.letter_buffer or new_letter != self.letter_buffer[-1]:
                                    self.letter_buffer.append(new_letter)
                                    print(f"Current buffer: {''.join(self.letter_buffer)}")
                                    self.last_detection_time = current_time
                                    
                                    self.suggestions = self.get_new_suggestions(self.letter_buffer)
                                    if self.suggestions:
                                        # If only one suggestion, lock it immediately
                                        if len(self.suggestions) == 1:
                                            self.current_word = self.suggestions[0]
                                            print(f"Word locked: {self.current_word}")
                                            self.reset_state()
                                        else:
                                            self.current_state = self.STATE_GESTURE_DETECTION
                                            print(f"Suggestions: {self.suggestions}")
            
            elif self.current_state == self.STATE_GESTURE_DETECTION:
                if current_time - self.last_gesture_time > self.gesture_cooldown:
                    gesture = self.detect_gesture(frame)
                    if gesture:
                        self.last_gesture_time = current_time
                        if gesture == "thumbs_up":
                            print(f"Accepted suggestions: {self.suggestions}")
                            self.possible_words = self.suggestions
                            self.waiting_for_next_letter = True
                            self.current_state = self.STATE_LETTER_DETECTION
                        elif gesture == "thumbs_down":
                            print("Rejected suggestions")
                            self.rejected_suggestions.update(self.suggestions)
                            self.suggestions = []
                            self.letter_buffer = []
                            self.current_state = self.STATE_LETTER_DETECTION
            
            self._draw_information(frame)
            return frame
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def reset_state(self):
        """Reset all state variables"""
        self.letter_buffer = []
        self.suggestions = []
        self.accepted_suggestions = []
        self.rejected_suggestions.clear()
        self.current_state = self.STATE_LETTER_DETECTION
        self.current_suggestion_index = 0
        self.last_gesture_time = time.time()

    def _draw_information(self, frame):
        # Display current state
        state_text = "GESTURE DETECTION" if self.current_state == self.STATE_GESTURE_DETECTION else "LETTER DETECTION"
        cv2.putText(frame, f"Mode: {state_text}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display letter buffer
        buffer_text = f"Letters: {''.join(self.letter_buffer)}"
        cv2.putText(frame, buffer_text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.waiting_for_next_letter and self.possible_words:
            cv2.putText(frame, f"Possible words: {', '.join(self.possible_words)}", 
                       (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Show next letter...", 
                       (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display suggestions and instructions
        elif self.current_state == self.STATE_GESTURE_DETECTION:
            cv2.putText(frame, "Suggestions:", 
                       (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            for i, suggestion in enumerate(self.suggestions):
                cv2.putText(frame, f"{i+1}. {suggestion}", 
                           (10, 390 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "Thumbs UP to accept, Thumbs DOWN to reject", 
                       (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display current word if completed
        if self.current_word:
            cv2.rectangle(frame, (10, 550), (400, 590), (0, 255, 0), 2)
            cv2.putText(frame, f"Locked Word: {self.current_word}", 
                       (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'C' to start new word", 
                       (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    def is_pose_stable(self, current_letter):
        
        current_time = time.time()
        
        # If pose changed, reset stability timer
        if current_letter != self.last_stable_pose:
            self.last_stable_pose = current_letter
            self.stable_pose_start_time = current_time
            return False
        
        # Check if pose has been held for long enough
        if self.stable_pose_start_time is None:
            self.stable_pose_start_time = current_time
            return False
            
        return (current_time - self.stable_pose_start_time) >= self.stable_pose_duration
    
    def detect_gesture(self, frame):
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                # Thumbs up
                if thumb_tip.y < wrist.y and thumb_tip.y < index_tip.y:
                    if time.time() - self.last_gesture_time > self.cooldown:
                        self.last_gesture_time = time.time()
                        return "thumbs_up"

                # Thumbs down
                if thumb_tip.y > wrist.y and thumb_tip.y > index_tip.y:
                    if time.time() - self.last_gesture_time > self.cooldown:
                        self.last_gesture_time = time.time()
                        return "thumbs_down"

        return None

        
    
def main():
    # Replace with your Hugging Face token
    HUGGINGFACE_TOKEN = "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"
    
    try:
        predictor = ASLWordPredictor(HUGGINGFACE_TOKEN)
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