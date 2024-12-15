import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from huggingface_hub import login
from test import ASLDetector

from joblib import load
from collections import deque
import statistics
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
        

class ASLAlphabetDetector:
    def __init__(self):
        self.model = load('alphabet_model.joblib')
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.prediction_window = deque(maxlen=10)
        self.current_letter = None

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
    HUGGINGFACE_TOKEN = "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"
    
    try:
        predictor = ASLAlphabetDetector(HUGGINGFACE_TOKEN)
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