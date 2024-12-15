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




class LlamaPredictor:
    def __init__(self, token):
        self.client = OpenAI(api_key=token)
    
        
        
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

#     
                    
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
    OPENAI_KEY= api_key # add api key in double quotes
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