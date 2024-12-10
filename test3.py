import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from huggingface_hub import login
from test2 import ASLDetector

# class LlamaPredictor:
#     def __init__(self, token):
#         # Login to Hugging Face with your token
#         login(token)
        
#         # Initialize tokenizer and model
#         try:
#             print("Loading tokenizer and model...")
#             # Let's use a different model that's more stable
#             model_name = "meta-llama/Llama-2-7b-chat-hf"  # Changed model
            
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_name,
#                 token=token
#             )
            
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 token=token,
#                 device_map="auto",
#                 torch_dtype=torch.float16  # Use half precision to save memory
#             )
#             print("Model loaded successfully!")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise
    
    # def get_suggestions(self, letters):
    #     if not letters:
    #         return []
            
    #     prompt = f"""Given the letters {''.join(letters)}, provide exactly 3 common English words that start with these letters.
    #     Output only the words separated by commas, nothing else."""
        
    #     try:
    #         inputs = self.tokenizer(prompt, return_tensors="pt")
    #         inputs = inputs.to(self.model.device)
            
    #         outputs = self.model.generate(
    #             **inputs,
    #             max_length=50,
    #             num_return_sequences=1,
    #             temperature=0.7,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.eos_token_id
    #         )
            
    #         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         # Extract just the words from the response
    #         try:
    #             words = response.split('\n')[-1].strip().split(',')
    #             return [word.strip() for word in words[:3]]
    #         except:
    #             return []
                
    #     except Exception as e:
    #         print(f"Error generating suggestions: {e}")
    #         return []

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
        self.detector = ASLDetector()
        self.letter_buffer = []
        self.current_word = ""
        self.suggestions = []
        self.last_detection_time = time.time()
        self.detection_cooldown = 1.5
        print("Initializing Llama model...")
        self.llama = LlamaPredictor(token)
        print("Model initialization complete")
        
    def process_frame(self, frame):
        frame = self.detector.process_frame(frame)
        current_time = time.time()
        
        if self.detector.current_letter:
            if current_time - self.last_detection_time > self.detection_cooldown:
                new_letter = self.detector.current_letter
                print(f"Detected letter: {new_letter}")
                
                # Only add if it's a new letter or buffer is empty
                if not self.letter_buffer or new_letter != self.letter_buffer[-1]:
                    self.letter_buffer.append(new_letter)
                    print(f"Current buffer: {''.join(self.letter_buffer)}")
                    self.last_detection_time = current_time
                    
                    print("Requesting suggestions...")
                    try:
                        self.suggestions = self.llama.get_suggestions(self.letter_buffer)
                        print(f"Received suggestions: {self.suggestions}")
                    except Exception as e:
                        print(f"Error getting suggestions: {e}")
        
        # Display buffer and suggestions
        buffer_text = f"Letters: {''.join(self.letter_buffer)}"
        cv2.putText(frame, buffer_text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for i, suggestion in enumerate(self.suggestions):
            cv2.putText(frame, f"Suggestion {i+1}: {suggestion}", 
                       (10, 350 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame



class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.last_gesture_time = 0  # Timestamp of the last detected gesture
        self.cooldown = 1.0  # Cooldown time in seconds

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
    cap = cv2.VideoCapture(0)
    detector = GestureDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        gesture = detector.detect_gesture(frame)
        if gesture:
            print(f"Detected Gesture: {gesture}")

        # Display the frame
        cv2.putText(frame, f"Gesture: {gesture if gesture else 'None'}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Thumbs Gesture Detection', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
