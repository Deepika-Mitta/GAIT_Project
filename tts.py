from elevenlabs import Voice, VoiceSettings, play, save
from elevenlabs.client import ElevenLabs

ELEVEN_API_KEY = "sk_a56a7da405cc3800f3b88624ebf52eb865336eb267e7e9ac"
VOICE_ID = "nPczCjzI2devNBz1zQrb"

client = ElevenLabs(api_key=ELEVEN_API_KEY)

audio = client.generate(
    text="Hello! My name is Arathi.",
    voice=Voice(
        voice_id=VOICE_ID
        #settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
    )
)

play(audio)

save(audio, "output_audio.mp3")

print("Audio file saved as output_audio.mp3")