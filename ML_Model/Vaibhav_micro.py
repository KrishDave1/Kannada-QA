import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from gtts import gTTS
import speech_recognition as sr
import tempfile
from io import BytesIO
import os

# Streamlit app title
st.title("Voice to MP3 Converter")

# Set sample rate and duration for recording
sample_rate = 44100  # Sample rate in Hz
duration = 20  # Duration in seconds

# Record function using sounddevice
def record_audio(duration, sample_rate):
    st.info("Recording for {} seconds...".format(duration))
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()  # Wait until recording is finished
    st.success("Recording completed!")
    return audio

# Function to transcribe audio
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Speech Recognition service error"

# Button to start recording
if st.button("Record and Convert"):
    # Record audio
    audio_data = record_audio(duration, sample_rate)
    # Save as WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
        write(temp_wav_file.name, sample_rate, audio_data)
        temp_wav_path = temp_wav_file.name

    # Transcribe the audio file
    transcription = transcribe_audio(temp_wav_path)
    st.write("Transcription: ", transcription)

    # Convert transcription to MP3
    tts = gTTS(text=transcription, lang='en')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    # Playback and download
    st.audio(mp3_fp, format="audio/mp3")
    st.download_button("Download MP3", mp3_fp, file_name="recorded_audio.mp3")

    # Clean up the temp WAV file
    os.remove(temp_wav_path)
