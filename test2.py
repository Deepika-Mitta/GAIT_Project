import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from huggingface_hub import login
from test import ASLDetector

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