import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
# from huggingface_hub import login
from test2 import ASLDetector
from elevenlabs import generate, play, Voice, set_api_key
from openai import OpenAI



# # class LlamaPredictor:
#     # def __init__(self, token):
#     #     login(token)
#     #     try:
#     #         print("Loading tokenizer and model...")
#     #         # Using a smaller model for faster responses
#     #         model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
#     #         self.tokenizer = AutoTokenizer.from_pretrained(
#     #             model_name,
#     #             token=token
#     #         )
            
#     #         self.model = AutoModelForCausalLM.from_pretrained(
#     #             model_name,
#     #             device_map="auto",
#     #             torch_dtype=torch.float16,
#     #             low_cpu_mem_usage=True
#     #         )
#     #         print("Model loaded successfully!")
#     #     except Exception as e:
#     #         print(f"Error loading model: {e}")
#     #         raise

#     # def get_suggestions(self, letters):
#     #     if not letters:
#     #         return []
            
#     #     try:
#     #         prefix = ''.join(letters)
#     #         print(f"Generating suggestions for prefix: {prefix}")
            
#     #         prompt = f"""<human>Generate 3 common English words that start with '{prefix}'. Reply with just the words separated by commas.</human>"""
            
#     #         # Set a timeout for generation
#     #         with torch.no_grad():
#     #             inputs = self.tokenizer(prompt, return_tensors="pt")
#     #             inputs = inputs.to(self.model.device)
                
#     #             print("Generating response...")
#     #             outputs = self.model.generate(
#     #                 **inputs,
#     #                 max_new_tokens=20,  # Shorter output
#     #                 num_return_sequences=1,
#     #                 temperature=0.7,
#     #                 do_sample=True,
#     #                 pad_token_id=self.tokenizer.eos_token_id,
#     #                 top_k=10,  # More focused sampling
#     #                 num_beams=1,  # No beam search for speed
#     #                 early_stopping=True
#     #             )
                
#     #             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#     #             print(f"Raw response: {response}")
                
#     #             # Clean up response
#     #             response = response.split('<assistant>')[-1].strip()
#     #             words = [word.strip() for word in response.split(',')]
#     #             valid_words = [w for w in words if w.lower().startswith(prefix.lower())]
#     #             print(f"Processed suggestions: {valid_words[:3]}")
#     #             return valid_words[:3]
                
#     #     except Exception as e:
#     #         print(f"Error in suggestion generation: {e}")
#     #         return []
        
class LlamaPredictor:
    def __init__(self, token):
        self.client = OpenAI(api_key=token)
        # login(token)
        # try:
        #     print("Loading tokenizer and model...")
        #     model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
        #     self.tokenizer = AutoTokenizer.from_pretrained(
        #         model_name,
        #         token=token
        #     )
            
        #     self.model = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         device_map="auto",
        #         torch_dtype=torch.float16,
        #         low_cpu_mem_usage=True
        #     )
            
            # Add common word list as fallback
            # self.common_words = {
            #     'A': ['APPLE', 'ABOUT', 'AFTER'],
            #     'B': ['BOOK', 'BETTER', 'BRING'],
            #     'C': ['CALL', 'COME', 'COOL'],
            #     'D': ['DOOR', 'DOWN', 'DARK'],
            #     'E': ['EARLY', 'EVERY', 'EVEN'],
            #     'F': ['FIND', 'FULL', 'FAST'],
            # }
        #     print("Model loaded successfully!")
        # except Exception as e:
        #     print(f"Error loading model: {e}")
        #     raise

    # def get_suggestions(self, letters):
    #     if not letters:
    #         return []
            
    #     try:
    #         prefix = ''.join(letters)
    #         print(f"Generating suggestions for prefix: {prefix}")
            
    #         prompt = f"<human>List 3 common English words that start with '{prefix}': </human>\n<assistant>"
            
    #         # Set a timeout for generation
    #         with torch.no_grad():
    #             inputs = self.tokenizer(prompt, return_tensors="pt")
    #             inputs = inputs.to(self.model.device)
                
    #             print("Generating response...")
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=20,  # Shorter output
    #                 num_return_sequences=1,
    #                 temperature=0.7,
    #                 do_sample=True,
    #                 pad_token_id=self.tokenizer.eos_token_id,
    #                 top_k=10,  # More focused sampling
    #                 num_beams=1,  # No beam search for speed
    #                 early_stopping=True
    #             )
                
    #             response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #             print(f"Raw response: {response}")
                
    #             # Improved response parsing
    #             try:
    #                 # Remove the prompt from response
    #                 response_text = response.split("<assistant>")[-1].strip()
                    
    #                 # Extract words using various delimiters
    #                 words = []
    #                 if ',' in response_text:
    #                     words = [word.strip() for word in response_text.split(',')]
    #                 else:
    #                     # If no commas, try splitting by spaces or newlines
    #                     words = [word.strip() for word in response_text.split()]
                    
    #                 # Filter valid words that start with prefix
    #                 valid_words = [word for word in words 
    #                              if word and word.upper().startswith(prefix.upper())][:3]
                    
    #                 print(f"Processed suggestions: {valid_words}")
    #                 return valid_words
                    
    #             except Exception as e:
    #                 print(f"Error parsing response: {e}")
    #                 return []
                
    #     except Exception as e:
    #         print(f"Error in suggestion generation: {e}")
    #         return []
    
        
        
    def get_suggestions(self, prefix):
        if not prefix:
            return []
            
        try:
            prefix_str = ''.join(prefix)
            print(f"Getting suggestions for prefix: {prefix_str}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates word suggestions."},
                    {"role": "user", "content": f"Give me exactly 3 common English words that start with '{prefix_str}'. Reply with just the words separated by commas, nothing else."}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            suggestions = response.choices[0].message.content.strip().split(',')
            suggestions = [s.strip() for s in suggestions]
            valid_suggestions = [s for s in suggestions if s.upper().startswith(prefix_str.upper())]
            
            print(f"Generated suggestions: {valid_suggestions}")
            return valid_suggestions[:3]
            
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []
        
# class ASLWordPredictor:
#     def __init__(self, token):
#         try:
#             # Basic initialization
#             self.detector = ASLDetector()
#             self.mp_hands = mp.solutions.hands
#             self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            
#             # State variables
#             self.STATE_LETTER_DETECTION = "letter_detection"
#             self.STATE_GESTURE_DETECTION = "gesture_detection"
#             self.current_state = self.STATE_LETTER_DETECTION
            
#             # Word tracking
#             self.letter_buffer = []
#             self.current_word = ""
#             self.suggestions = []
#             self.accepted_suggestions = []
#             self.rejected_suggestions = set()
#             self.current_suggestion_index = 0
#             self.collected_words = []
#             self.target_word_count = 3
#             self.sentence = None
            
#             # Timing and stability variables
#             self.last_detection_time = time.time()
#             self.last_gesture_time = 0
#             self.detection_cooldown = 1.5
#             self.stable_pose_duration = 1.0
#             self.gesture_cooldown = 2.0
#             self.cooldown = 1.0  # Added missing cooldown attribute
#             self.last_stable_pose = None
#             self.stable_pose_start_time = None
#             self.possible_words = []  # Add this to store accepted word list
#             self.waiting_for_next_letter = False  # Add this to track if we're waiting for next letter

#             self.tts_api_key = "sk_a56a7da405cc3800f3b88624ebf52eb865336eb267e7e9ac"
#             self.voice_id = "nPczCjzI2devNBz1zQrb"

#             print("Initializing Llama model...")
#             self.llama = LlamaPredictor(token)
#             print("Model initialization complete")
            
#         except Exception as e:
#             print(f"Error in initialization: {e}")
#             raise

    # def get_new_suggestions(self, prefix):
    #     """Get suggestions excluding previously rejected ones"""
    #     try:
    #         all_suggestions = self.llama.get_suggestions(prefix)
    #         return [s for s in all_suggestions if s not in self.rejected_suggestions]
    #     except Exception as e:
    #         print(f"Error getting suggestions: {e}")
    #         return []
        
#     def complete_word(self, word):
#         """Handle word completion and collection"""
#         print(f"Completed word: {word}")
#         self.collected_words.append(word)
#         print(f"Word {len(self.collected_words)}/{self.target_word_count} collected")
        
#         if len(self.collected_words) == self.target_word_count:
#             print("All words collected, generating sentence...")
#             self.generate_and_speak_sentence()
#         else:
#             print("Starting next word detection...")
#             self.reset_for_next_word()

#     def reset_for_next_word(self):
#         """Reset state while preserving collected words"""
#         self.letter_buffer = []
#         self.suggestions = []
#         self.accepted_suggestions = []
#         self.rejected_suggestions.clear()
#         self.current_state = self.STATE_LETTER_DETECTION
#         self.current_suggestion_index = 0
#         self.last_gesture_time = time.time()
#         self.current_word = ""
#         self.waiting_for_next_letter = False
#         self.possible_words = []

#     def generate_and_speak_sentence(self):
#         """Generate and speak a sentence using collected words"""
#         try:
#             words_str = ", ".join(self.collected_words)
#             prompt = f"""Task: Create a simple, natural sentence using EXACTLY these words: {words_str}.
#             Rules:
#             - Must use all words in any order
#             - Keep the sentence simple and natural
#             - Only provide the sentence, no explanations
#             Response:"""
            
#             with torch.no_grad():
#                 inputs = self.llama.tokenizer(prompt, return_tensors="pt",
#                                            padding=True, truncation=True, max_length=100)
#                 inputs = inputs.to(self.llama.model.device)
                
#                 outputs = self.llama.model.generate(
#                     **inputs,
#                     max_new_tokens=50,
#                     num_return_sequences=1,
#                     temperature=0.7,
#                     do_sample=True,
#                     pad_token_id=self.llama.tokenizer.eos_token_id,
#                     top_k=50,
#                     top_p=0.95
#                 )
                
#                 response = self.llama.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 self.sentence = response.split("Response:")[-1].strip()
                
#                 print(f"Generated sentence: {self.sentence}")
#                 self.play_audio(self.sentence)
                
#                 # Reset for next set of words
#                 self.collected_words = []
                
#         except Exception as e:
#             print(f"Error generating sentence: {e}")

    
#     def process_frame(self, frame):
#         try:
#             current_time = time.time()
            
#             # Show collection progress
#             cv2.putText(frame, f"Collecting word {len(self.collected_words) + 1}/{self.target_word_count}", 
#                     (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
#             if self.current_state == self.STATE_LETTER_DETECTION:
#                 frame = self.detector.process_frame(frame)
                
#                 if self.detector.current_letter:
#                     if self.stable_pose_start_time is not None:
#                         stability_progress = min(1.0, (current_time - self.stable_pose_start_time) / self.stable_pose_duration)
#                         progress_width = int(200 * stability_progress)
#                         cv2.rectangle(frame, (10, 270), (210, 290), (0, 0, 255), 2)
#                         cv2.rectangle(frame, (10, 270), (10 + progress_width, 290), (0, 255, 0), -1)
                    
#                     if self.is_pose_stable(self.detector.current_letter):
#                         if current_time - self.last_detection_time > self.detection_cooldown:
#                             new_letter = self.detector.current_letter
                            
#                             if not self.letter_buffer or new_letter != self.letter_buffer[-1]:
#                                 self.letter_buffer.append(new_letter)
#                                 print(f"Current buffer: {''.join(self.letter_buffer)}")
#                                 self.last_detection_time = current_time
                                
#                                 if self.waiting_for_next_letter and self.possible_words:
#                                     # Filter possible words based on next letter
#                                     current_pos = len(self.letter_buffer) - 1
#                                     matching_words = [word for word in self.possible_words 
#                                                 if len(word) > current_pos and 
#                                                 word[current_pos].upper() == new_letter]
                                    
#                                     if matching_words:
#                                         self.possible_words = matching_words
#                                         if len(matching_words) == 1:
#                                             self.current_word = matching_words[0]
#                                             print(f"Word locked: {self.current_word}")
#                                             self.play_audio(self.current_word)
#                                             self.complete_word(self.current_word)
#                                         else:
#                                             print(f"Possible words: {matching_words}")
#                                     else:
#                                         print("Letter doesn't match any words")
#                                         self.letter_buffer.pop()
#                                 else:
#                                     # Get new suggestions
#                                     self.suggestions = self.get_new_suggestions(self.letter_buffer)
#                                     if self.suggestions:
#                                         print(f"New suggestions: {self.suggestions}")
#                                         self.current_state = self.STATE_GESTURE_DETECTION
            
#             elif self.current_state == self.STATE_GESTURE_DETECTION:
#                 if current_time - self.last_gesture_time > self.gesture_cooldown:
#                     gesture = self.detect_gesture(frame)
#                     if gesture:
#                         self.last_gesture_time = current_time
#                         if gesture == "thumbs_up":
#                             print(f"Accepted suggestions: {self.suggestions}")
#                             if len(self.suggestions) == 1:
#                                 # Single suggestion - complete word
#                                 self.current_word = self.suggestions[0]
#                                 print(f"Word locked: {self.current_word}")
#                                 self.play_audio(self.current_word)
#                                 self.complete_word(self.current_word)
#                             else:
#                                 # Multiple suggestions - wait for next letter
#                                 self.possible_words = self.suggestions
#                                 self.waiting_for_next_letter = True
#                                 self.current_state = self.STATE_LETTER_DETECTION
#                         elif gesture == "thumbs_down":
#                             print("Rejected suggestions")
#                             self.rejected_suggestions.update(self.suggestions)
#                             self.letter_buffer = []
#                             self.current_state = self.STATE_LETTER_DETECTION
            
#             self._draw_information(frame)
#             return frame
                
#         except Exception as e:
#             print(f"Error in process_frame: {e}")
#             return frame
#     # def reset_state(self):
#     #     """Reset all state variables"""
#     #     self.letter_buffer = []
#     #     self.suggestions = []
#     #     self.accepted_suggestions = []
#     #     self.rejected_suggestions.clear()
#     #     self.current_state = self.STATE_LETTER_DETECTION
#     #     self.current_suggestion_index = 0
#     #     self.last_gesture_time = time.time()

#     # def _draw_information(self, frame):
#         # # Display current state
#         # state_text = "GESTURE DETECTION" if self.current_state == self.STATE_GESTURE_DETECTION else "LETTER DETECTION"
#         # cv2.putText(frame, f"Mode: {state_text}", 
#         #            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
#         # # Display letter buffer
#         # buffer_text = f"Letters: {''.join(self.letter_buffer)}"
#         # cv2.putText(frame, buffer_text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         # if self.waiting_for_next_letter and self.possible_words:
#         #     cv2.putText(frame, f"Possible words: {', '.join(self.possible_words)}", 
#         #                (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         #     cv2.putText(frame, "Show next letter...", 
#         #                (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # # Display suggestions and instructions
#         # elif self.current_state == self.STATE_GESTURE_DETECTION:
#         #     cv2.putText(frame, "Suggestions:", 
#         #                (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         #     for i, suggestion in enumerate(self.suggestions):
#         #         cv2.putText(frame, f"{i+1}. {suggestion}", 
#         #                    (10, 390 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         #     cv2.putText(frame, "Thumbs UP to accept, Thumbs DOWN to reject", 
#         #                (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         # # Display current word if completed
#         # if self.current_word:
#         #     cv2.rectangle(frame, (10, 550), (400, 590), (0, 255, 0), 2)
#         #     cv2.putText(frame, f"Locked Word: {self.current_word}", 
#         #                (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         #     cv2.putText(frame, "Press 'C' to start new word", 
#         #                (20, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
#         # if self.collected_words:
#         #     y_pos = 650
#         #     cv2.putText(frame, "Collected words:", 
#         #                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         #     for i, word in enumerate(self.collected_words):
#         #         cv2.putText(frame, f"{i+1}. {word}", 
#         #                    (20, y_pos + 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 
#         #                    0.7, (0, 255, 0), 2)
            
#     def _draw_information(self, frame):
#         # Display current state
#         state_text = "GESTURE DETECTION" if self.current_state == self.STATE_GESTURE_DETECTION else "LETTER DETECTION"
#         cv2.putText(frame, f"Mode: {state_text}", 
#                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
#         # Display letter buffer
#         buffer_text = f"Letters: {''.join(self.letter_buffer)}"
#         cv2.putText(frame, buffer_text, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         # Make suggestions more prominent
#         if self.current_state == self.STATE_GESTURE_DETECTION:
#             # Draw background box for suggestions
#             overlay = frame.copy()
#             cv2.rectangle(overlay, (5, 320), (400, 530), (0, 0, 0), -1)
#             cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
#             cv2.putText(frame, "WORD SUGGESTIONS:", 
#                     (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             for i, suggestion in enumerate(self.suggestions):
#                 cv2.putText(frame, f"{i+1}. {suggestion}", 
#                         (30, 390 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(frame, "👍 THUMBS UP: Accept suggestions", 
#                     (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             cv2.putText(frame, "👎 THUMBS DOWN: Get new suggestions", 
#                     (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
#     def is_pose_stable(self, current_letter):
        
#         current_time = time.time()
        
#         # If pose changed, reset stability timer
#         if current_letter != self.last_stable_pose:
#             self.last_stable_pose = current_letter
#             self.stable_pose_start_time = current_time
#             return False
        
#         # Check if pose has been held for long enough
#         if self.stable_pose_start_time is None:
#             self.stable_pose_start_time = current_time
#             return False
            
#         return (current_time - self.stable_pose_start_time) >= self.stable_pose_duration
    
#     def detect_gesture(self, frame):
        
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb_frame)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
#                 index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

#                 # Thumbs up
#                 if thumb_tip.y < wrist.y and thumb_tip.y < index_tip.y:
#                     if time.time() - self.last_gesture_time > self.cooldown:
#                         self.last_gesture_time = time.time()
#                         return "thumbs_up"

#                 # Thumbs down
#                 if thumb_tip.y > wrist.y and thumb_tip.y > index_tip.y:
#                     if time.time() - self.last_gesture_time > self.cooldown:
#                         self.last_gesture_time = time.time()
#                         return "thumbs_down"

#         return None

#     def play_audio(self, word):
#         if not self.tts_api_key:
#             print(f"Word detected: {word}")
#             return
        
#         try:
#             set_api_key(self.tts_api_key)
#             audio = generate(
#                 text=word,
#                 voice=self.voice_id  
#             )
#             play(audio)
#         except Exception as e:
#             print(f"Error in generating TTS audio: {e}")

        
    
# def main():
#     # Replace with your Hugging Face token
#     HUGGINGFACE_TOKEN = "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"
    
#     try:
#         predictor = ASLWordPredictor(HUGGINGFACE_TOKEN)
#         cap = cv2.VideoCapture(0)
        
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 continue
                
#             frame = predictor.process_frame(frame)
#             cv2.imshow('ASL Word Prediction', frame)
            
#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('q'):
#                 break
#             elif key == ord('c'):  # Clear buffer
#                 predictor.letter_buffer = []
#                 predictor.suggestions = []
#             elif key == ord(' '):  # Space to accept word
#                 if predictor.suggestions:
#                     predictor.current_word = predictor.suggestions[0]
#                     print(f"Accepted word: {predictor.current_word}")
#                     predictor.letter_buffer = []
#                     predictor.suggestions = []
                
#         cap.release()
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

class ASLWordPredictor:
    def __init__(self, token):
        self.client = OpenAI(api_key=token)
        # Initialize components
        self.detector = ASLDetector()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.llama = LlamaPredictor(token)

        
        # State tracking
        self.current_state = "LETTER_DETECTION"  # or "GESTURE_DETECTION"
        self.letter_buffer = []
        self.current_word = ""
        self.collected_words = []
        self.target_word_count = 3
        self.should_exit = False
        self.sentence_played = False
        
        # Suggestion handling
        self.current_suggestions = []
        self.possible_words = []
        self.waiting_for_next_letter = False
        self.rejected_suggestions = set()
        
        # Timing and stability
        self.last_detection_time = time.time()
        self.last_gesture_time = time.time()
        self.detection_cooldown = 1.5
        self.gesture_cooldown = 2.0
        self.stable_pose_duration = 1.0
        self.last_stable_pose = None
        self.stable_pose_start_time = None
        
        # TTS setup
        self.tts_api_key = "sk_a56a7da405cc3800f3b88624ebf52eb865336eb267e7e9ac"
        self.voice_id = "nPczCjzI2devNBz1zQrb"
        set_api_key(self.tts_api_key)

    def get_new_suggestions(self, prefix):
        """Get suggestions excluding previously rejected ones"""
        try:
            all_suggestions = self.llama.get_suggestions(prefix)
            return [s for s in all_suggestions if s not in self.rejected_suggestions]
        except Exception as e:
            print(f"Error getting suggestions: {e}")
            return []

    # def process_frame(self, frame):
    #     try:
    #         current_time = time.time()
            
    #         # Show collection progress
    #         cv2.putText(frame, f"Collecting word {len(self.collected_words) + 1}/{self.target_word_count}", 
    #                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    #         if self.current_state == "LETTER_DETECTION":
    #             # Process frame for letter detection
    #             frame = self.detector.process_frame(frame)
    #             detected_letter = self.detector.current_letter

    #             if detected_letter:
    #                 # Show stability progress bar
    #                 if self.stable_pose_start_time:
    #                     stability_progress = min(1.0, (current_time - self.stable_pose_start_time) / self.stable_pose_duration)
    #                     progress_width = int(200 * stability_progress)
    #                     cv2.rectangle(frame, (10, 60), (210, 80), (0, 0, 255), 2)
    #                     cv2.rectangle(frame, (10, 60), (10 + progress_width, 80), (0, 255, 0), -1)

    #                 if self.is_pose_stable(detected_letter):
    #                     if current_time - self.last_detection_time > self.detection_cooldown:
    #                         self.handle_letter_detection(detected_letter)
    #                         self.last_detection_time = current_time

    #         elif self.current_state == "GESTURE_DETECTION":
    #             if current_time - self.last_gesture_time > self.gesture_cooldown:
    #                 gesture = self.detect_gesture(frame)
    #                 if gesture:
    #                     self.handle_gesture(gesture)
    #                     self.last_gesture_time = current_time

    #         # Update display
    #         self.draw_interface(frame)
    #         return frame

    #     except Exception as e:
    #         print(f"Error in process_frame: {e}")
    #         return frame

    def process_frame(self, frame):
        try:
            current_time = time.time()
            
            # Show collection progress
            cv2.putText(frame, f"Collecting word {len(self.collected_words) + 1}/{self.target_word_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if self.current_state == "LETTER_DETECTION":
                # Process frame for letter detection
                frame = self.detector.process_frame(frame)
                detected_letter = self.detector.current_letter

                if detected_letter:
                    # Show stability progress bar
                    if self.stable_pose_start_time:
                        stability_progress = min(1.0, (current_time - self.stable_pose_start_time) / self.stable_pose_duration)
                        progress_width = int(200 * stability_progress)
                        cv2.rectangle(frame, (10, 60), (210, 80), (0, 0, 255), 2)
                        cv2.rectangle(frame, (10, 60), (10 + progress_width, 80), (0, 255, 0), -1)

                    if self.is_pose_stable(detected_letter):
                        if current_time - self.last_detection_time > self.detection_cooldown:
                            self.handle_letter_detection(detected_letter)
                            self.last_detection_time = current_time

            elif self.current_state == "GESTURE_DETECTION":
                if current_time - self.last_gesture_time > self.gesture_cooldown:
                    gesture = self.detect_gesture(frame)
                    if gesture:
                        self.handle_gesture(gesture)
                        self.last_gesture_time = current_time
            
            # Check if we've collected all words and need to generate sentence
            if len(self.collected_words) == self.target_word_count:
                # Generate and speak sentence
                sentence = self.predictor.generate_sentence(self.collected_words)
                if sentence:
                    print(f"Final sentence: {sentence}")
                    # Display sentence on frame
                    cv2.putText(frame, "Generated Sentence:", 
                            (10, frame.shape[0] - 120), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                    
                    # Split long sentences into multiple lines
                    words = sentence.split()
                    line = ""
                    y_position = frame.shape[0] - 80
                    for word in words:
                        if len(line + word) < 40:  # Adjust based on frame width
                            line += word + " "
                        else:
                            cv2.putText(frame, line, 
                                    (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
                            y_position += 30
                            line = word + " "
                    if line:
                        cv2.putText(frame, line, 
                                (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, (0, 255, 0), 2)
                    
                    # Play audio
                    self.play_audio(sentence)
                    
                    # Reset collection
                    self.collected_words = []
                    self.current_state = "LETTER_DETECTION"
                    print("Starting new word collection")

            # Update display
            self.draw_interface(frame)
            return frame

        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame

    def handle_letter_detection(self, letter):
        if not self.letter_buffer or letter != self.letter_buffer[-1]:
            self.letter_buffer.append(letter)
            print(f"Letter buffer: {''.join(self.letter_buffer)}")

            if self.waiting_for_next_letter and self.possible_words:
                # Check if letter matches any possible words
                current_pos = len(self.letter_buffer) - 1
                matching_words = [word for word in self.possible_words 
                                if len(word) > current_pos and 
                                word[current_pos].upper() == letter]
                
                if matching_words:
                    self.possible_words = matching_words
                    if len(matching_words) == 1:
                        # We found our word!
                        self.complete_word(matching_words[0])
                    else:
                        print(f"Possible words narrowed to: {matching_words}")
                else:
                    print("Letter doesn't match any words")
                    self.letter_buffer.pop()
            else:
                # Get new suggestions
                self.current_suggestions = self.get_new_suggestions(self.letter_buffer)
                if self.current_suggestions:
                    self.current_state = "GESTURE_DETECTION"

    def handle_gesture(self, gesture):
        if gesture == "thumbs_up":
            if len(self.current_suggestions) == 1:
                self.complete_word(self.current_suggestions[0])
            else:
                self.possible_words = self.current_suggestions
                self.waiting_for_next_letter = True
                self.current_state = "LETTER_DETECTION"
        elif gesture == "thumbs_down":
            self.rejected_suggestions.update(self.current_suggestions)
            self.letter_buffer = []
            self.current_suggestions = []
            self.current_state = "LETTER_DETECTION"

    def complete_word(self, word):
        print(f"Completed word: {word}")
        self.play_audio(word)
        self.collected_words.append(word)
        
        if len(self.collected_words) == self.target_word_count:
            self.generate_and_speak_sentence()
            self.collected_words = []
        
        self.reset_word_state()

    def reset_word_state(self):
        self.letter_buffer = []
        self.current_suggestions = []
        self.possible_words = []
        self.waiting_for_next_letter = False
        self.current_state = "LETTER_DETECTION"
        self.rejected_suggestions.clear()

#     def generate_and_speak_sentence(self):
#         try:
#             words_str = ", ".join(self.collected_words)
#             print(f"Generating sentence using words: {words_str}")
            
#             # Much simpler, more direct prompt
#             prompt = f"""<human>The words are: {words_str}
# Complete this: The</human>
# <assistant>The"""

#             with torch.no_grad():
#                 inputs = self.llama.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
#                 inputs = inputs.to(self.llama.model.device)
                
#                 outputs = self.llama.model.generate(
#                     **inputs,
#                     max_new_tokens=30,
#                     temperature=0.7,
#                     do_sample=True,
#                     top_k=50,
#                     top_p=0.9,
#                     repetition_penalty=1.2,
#                     early_stopping=False
#                 )
                
#                 response = self.llama.tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 print(f"Raw model response: {response}")
                
#                 try:
#                     # Extract the sentence, starting with "The"
#                     sentence = response.split("The")[-1].strip()
#                     sentence = "The" + sentence
                    
#                     print(f"Extracted sentence: {sentence}")
                    
#                     # Verify all words are present (case-insensitive)
#                     sentence_lower = sentence.lower()
#                     missing_words = []
#                     for word in self.collected_words:
#                         if word.lower() not in sentence_lower:
#                             missing_words.append(word)
                    
#                     if not missing_words:
#                         print(f"Success! Generated sentence: {sentence}")
#                         self.play_audio(sentence)
                
#                 except Exception as e:
#                     print(f"Error processing sentence: {e}")
#                     fallback = f"The {self.collected_words[0]} could {self.collected_words[1]} maintain its {self.collected_words[2]}."
#                     print(f"Using error fallback: {fallback}")
#                     self.play_audio(fallback)
                    
#         except Exception as e:
#             print(f"Error in sentence generation: {e}")
                    
    def generate_and_speak_sentence(self):
        try:
            words_str = ", ".join(self.collected_words)
            print(f"Generating sentence using words: {words_str}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates simple, natural sentences."},
                    {"role": "user", "content": f"Create a simple, natural sentence using exactly these words: {words_str}. Reply with just the sentence, nothing else."}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            sentence = response.choices[0].message.content.strip()
            print(f"Generated sentence: {sentence}")
            self.play_audio(sentence)
            self.sentence_played = True
            self.should_exit = True
            return sentence
            
        except Exception as e:
            print(f"Error generating sentence: {e}")
            return None

    def draw_interface(self, frame):
        # Show current mode
        cv2.putText(frame, f"Mode: {self.current_state}", 
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show letter buffer
        cv2.putText(frame, f"Letters: {''.join(self.letter_buffer)}", 
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show suggestions or possible words
        if self.current_state == "GESTURE_DETECTION":
            cv2.putText(frame, "Suggestions:", 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            for i, suggestion in enumerate(self.current_suggestions):
                cv2.putText(frame, f"{suggestion}", 
                           (10, 240 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "👍 Accept  👎 Reject", 
                       (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show collected words
        y_pos = 360
        cv2.putText(frame, "Collected words:", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for i, word in enumerate(self.collected_words):
            cv2.putText(frame, f"{i+1}. {word}", 
                       (10, y_pos + 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def is_pose_stable(self, current_letter):
        current_time = time.time()
        
        if current_letter != self.last_stable_pose:
            self.last_stable_pose = current_letter
            self.stable_pose_start_time = current_time
            return False
            
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
                
                if thumb_tip.y < wrist.y and thumb_tip.y < index_tip.y:
                    return "thumbs_up"
                elif thumb_tip.y > wrist.y and thumb_tip.y > index_tip.y:
                    return "thumbs_down"
        
        return None

    def play_audio(self, text):
        try:
            audio = generate(text=text, voice=self.voice_id)
            play(audio)
        except Exception as e:
            print(f"Error in TTS: {e}")

def main():
    # HUGGINGFACE_TOKEN = "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"
    OPENAI_KEY= api_key
    predictor = ASLWordPredictor(OPENAI_KEY)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = predictor.process_frame(frame)
        cv2.imshow('ASL Word Predictor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()