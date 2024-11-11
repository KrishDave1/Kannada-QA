import os
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
data = load_text_files(r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data")

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

# File uploader for pre-recorded audio file input
audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

def close_browser():
    st.write(
        """<script>
        setTimeout(function() { window.close(); }, 5000);
        </script>""",
        unsafe_allow_html=True,
    )

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
        st.write("Finding the best answer from context...")
        answer = get_best_answer(question, data)
        st.write(f"Answer: {answer}")

        # Display final message, close app, and close browser tab
        st.write("Thanks for using this")
        time.sleep(5)
        st.stop()  # Stops the Streamlit app
        os._exit(0)  # Terminates the command-line process
        close_browser()  # Injects JavaScript to close the browser tab

    # Clean up temporary file
    os.remove(temp_audio_path)

# Option to record audio using the microphone
st.write("Or, use your microphone to ask a question:")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.audio_frames.append(audio)
        return frame

# Start webrtc_streamer for audio recording
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Adjusted RTC configuration
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

if st.button("Process Recorded Audio"):
    if webrtc_ctx.audio_processor and webrtc_ctx.audio_processor.audio_frames:
        # Concatenate audio frames and save to a temporary file
        audio_data = np.concatenate(webrtc_ctx.audio_processor.audio_frames, axis=0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f, audio_data, 16000)
            temp_audio_path = f.name

        # Transcribe audio
        st.write("Transcribing audio...")
        question = transcribe_audio(temp_audio_path)
        if question:
            st.write(f"Transcribed Question: {question}")

            # Get the best answer
            st.write("Finding the best answer from context...")
            answer = get_best_answer(question, data)
            st.write(f"Answer: {answer}")

            # Display final message, close app, and close browser tab
            st.write("Thanks for using this")
            time.sleep(5)
            st.stop()  # Stops the Streamlit app
            os._exit(0)  # Terminates the command-line process
            close_browser()  # Injects JavaScript to close the browser tab

        # Clean up temporary file
        os.remove(temp_audio_path)
    else:
        st.write("Please record audio first.")