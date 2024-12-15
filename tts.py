from elevenlabs import generate, save

text = "Hello! My name is Arathi."
voice_id = "nPczCjzI2devNBz1zQrb"
api_key = "sk_bfdf6f56d87e5bd866564d11f1c2b819e5d5998020337db1"

try:
    audio = generate(
        text=text,
        api_key=api_key,
        voice=voice_id
    )
    
    save(audio, "output_audio.mp3")
    print("Audio file saved successfully!")
except Exception as e:
    print(f"Error: {e}")