import os
import json
import whisper
from transformers import pipeline
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import tempfile
import numpy as np
import soundfile as sf
import time

# Load the Whisper model
model_m = whisper.load_model("medium")

# Initialize the QA pipeline
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Function to load the knowledge base
def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Function to save updates to the knowledge base
def save_knowledge_base(knowledge_base):
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

# Load the knowledge base
knowledge_base = load_knowledge_base()

# Function to load text files and combine into a context
def load_text_files(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            data.append(file.read())
    return data

# Load all contexts from text files
data = load_text_files(
    r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"
)

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    result = model_m.transcribe(file_path)
    return result["text"]

# Function to get the best answer from the knowledge base or context
def get_best_answer(question, contexts, knowledge_base):
    # Check if the question exists in the knowledge base
    if question in knowledge_base:
        return knowledge_base[question]

    # If not in the knowledge base, use the QA pipeline
    best_answer = ""
    best_score = 0
    for context in contexts:
        result = qa_pipeline(question=question, context=context)
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer

# Define the Streamlit frontend
st.title("Audio Question Answering System with Feedback")

# File uploader for pre-recorded audio file input
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_file.getbuffer())
        temp_audio_path = f.name

    # Convert speech to text
    st.write("Transcribing audio...")
    question = transcribe_audio(temp_audio_path)
    if question:
        st.write(f"Transcribed Question: {question}")

        # Get the best answer
        st.write("Finding the best answer from context or knowledge base...")
        answer = get_best_answer(question, data, knowledge_base)
        st.write(f"Answer: {answer}")

        # Feedback mechanism
        st.write("Was this answer correct?")
        if st.button("Yes"):
            st.write("Thank you for confirming!")
        elif st.button("No"):
            correct_answer = st.text_input("Please provide the correct answer:")
            if st.button("Submit Correction"):
                knowledge_base[question] = correct_answer
                save_knowledge_base(knowledge_base)
                st.write("Thank you! The answer has been updated.")

        # Clean up temporary file
        os.remove(temp_audio_path)

# Option to record audio using the microphone
st.write("Or, use your microphone to ask a question:")
