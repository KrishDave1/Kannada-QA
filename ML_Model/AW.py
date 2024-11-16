import os
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import streamlit as st
import whisper
from transformers import pipeline

# Step 1: Load the Text File as a Knowledge Base
def load_knowledge_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Step 2: Initialize the QA Model
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Step 3: Function to Find the Best Answer in the Text File
def get_best_answer(question, context):
    try:
        # Use the QA pipeline to find the answer in the context
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"Could not find a specific answer. Please refine your question. ({str(e)})"

# Step 4: Function to Record Audio
def record_audio(duration, sample_rate):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    st.success("Recording completed!")
    return audio

# Step 5: Function to Transcribe Audio Using Whisper
model_m = whisper.load_model("medium")

def transcribe_audio(file_path):
    result = model_m.transcribe(file_path)
    return result["text"]

# Step 6: Main Function
def main():
    # Path to the text file containing all knowledge
    knowledge_file_path = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\combined_output1.txt"  # Replace with the path to your file
    context = load_knowledge_file(knowledge_file_path)

    # Streamlit interface
    st.title("Audio Question Answering System")
    sample_rate = 44100  # Sample rate in Hz
    duration = 15  # Duration in seconds

    st.write("You can either record your question or type it directly.")

    # Record and process audio
    if st.button("Record and Process Audio"):
        # Record audio
        audio_data = record_audio(duration, sample_rate)

        # Save the recorded audio as a WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            write(temp_wav_file.name, sample_rate, audio_data)
            temp_audio_path = temp_wav_file.name

        # Transcribe the audio to text using Whisper
        st.write("Transcribing audio...")
        question = transcribe_audio(temp_audio_path)
        st.write(f"Transcribed Question: {question}")

        # Get the best answer from the text file
        answer = get_best_answer(question, context)
        st.write(f"Answer: {answer}")

        # Clean up temporary files
        os.remove(temp_audio_path)

    # Option for direct text input as a question
    question = st.text_input("Or, type your question directly:")
    if question:
        # Get the best answer from the text file
        answer = get_best_answer(question, context)
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
