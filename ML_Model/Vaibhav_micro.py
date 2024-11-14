import os
import whisper
from transformers import pipeline
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import sounddevice as sd
from scipy.io.wavfile import write
from gtts import gTTS
import tempfile
import numpy as np
import soundfile as sf
from io import BytesIO

# Load the Whisper model
model_m = whisper.load_model("medium")

# Define the question-answering pipeline
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Function to load text files and combine them into a single context
def load_text_files(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            data.append(file.read())
    return data

# Load all contexts from the text files
data = load_text_files(r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data")

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    result = model_m.transcribe(file_path)
    return result["text"]

# Function to find the best answer from the context
def get_best_answer(question, contexts):
    best_answer = ""
    best_score = 0

    for context in contexts:
        result = qa_pipeline(question=question, context=context)
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer

# Define the Streamlit frontend
st.title("Audio Question Answering System")

# Record audio function
def record_audio(duration, sample_rate):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    st.success("Recording completed!")
    return audio

# Button to record, convert to MP3, and process
if st.button("Record and Process Audio"):
    sample_rate = 44100  # Sample rate in Hz
    duration = 15  # Duration in seconds
    
    # Record audio
    audio_data = record_audio(duration, sample_rate)
    
    # Save as WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        write(temp_wav_file.name, sample_rate, audio_data)
        temp_wav_path = temp_wav_file.name

    # Convert WAV to MP3
    with open(temp_wav_path, "rb") as f:
        question = transcribe_audio(temp_wav_path)
        st.write(f"Transcribed Question: {question}")

        # Get the best answer
        st.write("Finding the best answer from context...")
        answer = get_best_answer(question, data)
        st.write(f"Answer: {answer}")

        # Convert transcription to MP3 for download
        tts = gTTS(text=answer, lang="en")
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        # Playback and download
        st.audio(mp3_fp, format="audio/mp3")
        st.download_button("Download MP3 Answer", mp3_fp, file_name="answer.mp3")

    # Clean up temporary files
    os.remove(temp_wav_path)
