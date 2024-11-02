api_key = sk-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

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

