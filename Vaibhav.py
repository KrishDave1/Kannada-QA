import openai
import os

# Set your API key
openai.api_key = "api_key"

def transcribe_with_whisper(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

audio_dir = r "C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\dataset
transcriptions = {}

for file in os.listdir(audio_dir):
    if file.endswith(".mp3"):
        file_path = os.path.join(audio_dir, file)
        transcription = transcribe_with_whisper(file_path)
        transcriptions[file] = transcription
        print(f"Transcribed {file}: {transcription}")

