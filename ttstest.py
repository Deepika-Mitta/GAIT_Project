from elevenlabs import text_to_Speech, play, save
import os

# Set your API key
api_key = os.getenv("MyTextToSpeechKey")

# Text to convert to speech
text = "Hello! This is a sample text-to-speech conversion using ElevenLabs."

# Generate speech
audio = text_to_Speech(
    text=text,
    api_key=api_key,
    voice="Rachel"  # Replace with other available voices
)

# Play the audio
play(audio)

# Save the audio to a file
save(audio, "output_audio.mp3")

print("Audio file saved as output_audio.mp3")

