import torch
import whisper
import librosa

# Load the medium model
model_m = whisper.load_model("medium")

# File path to your audio file
file_path = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\dataset\audiocorpus\SandalWoodNewsStories_297.mp3"

# Load and resample audio to 16kHz
audio, sr = librosa.load(file_path, sr=16000)
whisper_audio = torch.tensor(audio)

# Set a larger chunk duration for fewer segments
chunk_duration = 60  # Increased from 30 to 60 seconds
transcription_result = []  # To store each chunk's transcription

# Detect language once on a sample of the audio
sample_audio = whisper.pad_or_trim(torch.tensor(audio[:sr * chunk_duration]))
mel = whisper.log_mel_spectrogram(sample_audio).to(model_m.device)
_, probs = model_m.detect_language(mel)
detected_language = max(probs, key=probs.get)
print(f"Detected language: {detected_language}")

# Process audio in larger chunks with reduced beam size for faster decoding
options = whisper.DecodingOptions(beam_size=2, temperature=0.7, fp16=False)  # Smaller beam size

for i in range(0, len(audio), chunk_duration * sr):
    # Process the larger chunk
    chunk = audio[i:i + chunk_duration * sr]
    whisper_audio = whisper.pad_or_trim(torch.tensor(chunk))
    
    # Generate mel spectrogram and decode with faster settings
    mel = whisper.log_mel_spectrogram(whisper_audio).to(model_m.device)
    result = whisper.decode(model_m, mel, options)
    print(f"Chunk {i // (chunk_duration * sr) + 1} transcription:", result.text)
    
    # Append the chunk's transcription
    transcription_result.append(result.text)

# Combine all chunks' transcriptions into one final transcription
full_transcription = " ".join(transcription_result)
print("Full Transcription:", full_transcription)

# Save the full transcription to a text file
output_file_path = "kannada_transcription.txt"
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(full_transcription)
print(f"Kannada transcription saved to {output_file_path}")

# Translate the entire audio to English
translation_result = model_m.transcribe(file_path, language="en", fp16=False)["text"]
print("Translation to English:", translation_result)
