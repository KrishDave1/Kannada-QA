import torch
import whisper
import librosa

# Load the medium model
model_m = whisper.load_model("medium")

# File path to your audio file
file_path = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\dataset\audiocorpus\SandalWoodNewsStories_249.mp3"

# Load audio and resample to 16kHz
audio, sr = librosa.load(file_path, sr=16000)
whisper_audio = torch.tensor(audio)

# Parameters for segmenting the audio (in seconds)
chunk_duration = 30  # Duration for each audio chunk
transcription_result = []  # To store each chunk's transcription

# Process the audio in chunks
for i in range(0, len(audio), chunk_duration * sr):
    # Extract the chunk and apply padding if necessary
    chunk = audio[i:i + chunk_duration * sr]
    whisper_audio = whisper.pad_or_trim(torch.tensor(chunk))

    # Generate the mel spectrogram for the chunk
    mel = whisper.log_mel_spectrogram(whisper_audio).to(model_m.device)
    
    # Detect language for the chunk
    _, probs = model_m.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"Detected language for chunk {i // (chunk_duration * sr) + 1}: {detected_language}")

    # Set decoding options with beam search
    options = whisper.DecodingOptions(beam_size=5, temperature=0.5, fp16=False)

    # Decode the audio chunk and print the transcription
    result = whisper.decode(model_m, mel, options)
    print(f"Chunk {i // (chunk_duration * sr) + 1} transcription:", result.text)
    
    # Append the chunk's transcription to the list
    transcription_result.append(result.text)

# Combine all chunks' transcriptions into one final transcription
full_transcription = " ".join(transcription_result)
print("Full Transcription:", full_transcription)

# Translate the entire audio to English
translation_result = model_m.transcribe(file_path, language="en", fp16=False)["text"]
print("Translation to English:", translation_result)
