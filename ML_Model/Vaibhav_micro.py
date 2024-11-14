import os
import json
import sounddevice as sd
from scipy.io.wavfile import write
from gtts import gTTS
import tempfile
from io import BytesIO
import numpy as np
import soundfile as sf
import whisper
from transformers import pipeline
import streamlit as st

# Step 1: Load and Combine Text Files
def load_text_files(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            data.append(file.read())
    return data


# Step 2: Initialize the QA Model
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Step 3: Load Knowledge Base
def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Step 4: Save Knowledge Base
def save_knowledge_base(knowledge_base):
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)

# Step 5: Function to Find Best Answer Across Files
def get_best_answer(question, contexts, knowledge_base):
    # Check if an updated answer is already available in the knowledge base
    if question in knowledge_base:
        return knowledge_base[question]

    best_answer = ""
    best_score = 0
    for context in contexts:
        result = qa_pipeline(question=question, context=context)
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer

# Step 6: Function to Record Audio
def record_audio(duration, sample_rate):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    st.success("Recording completed!")
    return audio

# Step 7: Function to Transcribe Audio Using Whisper
model_m = whisper.load_model("medium")

def transcribe_audio(file_path):
    result = model_m.transcribe(file_path)
    return result["text"]

# Step 8: Main Function with Feedback Mechanism and Audio Handling
def main():
    # Load text data and knowledge base
    directory_path = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"
    data = load_text_files(directory_path)
    knowledge_base = load_knowledge_base()

    # Streamlit interface
    st.title("Audio Question Answering System")

    sample_rate = 44100  # Sample rate in Hz
    duration = 15  # Duration in seconds

    # Option to record or ask a question
    st.write("You can either ask a question or record your question using the microphone.")
    
    # Record and process audio
    if st.button("Record and Process Audio"):
        # Record audio
        audio_data = record_audio(duration, sample_rate)
        
        # Save the recorded audio as WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            write(temp_wav_file.name, sample_rate, audio_data)
            temp_audio_path = temp_wav_file.name

        # Transcribe the audio to text using Whisper
        st.write("Transcribing audio...")
        question = transcribe_audio(temp_audio_path)
        st.write(f"Transcribed Question: {question}")

        # Get the best answer from the knowledge base
        answer = get_best_answer(question, data, knowledge_base)
        st.write(f"Answer: {answer}")

        # Get feedback from the user to improve the knowledge base
        feedback = st.text_input("Was this answer correct? If not, provide the correct answer.")
        if feedback:
            knowledge_base[question] = feedback
            save_knowledge_base(knowledge_base)
            st.success("Answer updated in knowledge base!")

        # Clean up temporary files
        os.remove(temp_audio_path)

    # Option for direct text input as a question
    else:
        question = st.text_input("Or, enter your question directly:")
        if question:
            # Get the answer for the direct question
            answer = get_best_answer(question, data, knowledge_base)
            st.write(f"Answer: {answer}")

            # Get feedback from the user to improve the knowledge base
            feedback = st.text_input("Was this answer correct? If not, provide the correct answer.")
            if feedback:
                knowledge_base[question] = feedback
                save_knowledge_base(knowledge_base)
                st.success("Answer updated in knowledge base!")

if __name__ == "__main__":
    main()
