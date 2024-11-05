
import torch
import whisper
import pytube
import librosa
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd

model_m = whisper.load_model("medium") #Load the medium size model

file_path=r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\dataset\audiocorpus\SandalWoodNewsStories_23.mp3"
audio = whisper.load_audio(file_path) #Load the audio file
audio = whisper.pad_or_trim(audio) #Pad or trim the audio to have a fixed length
mel = whisper.log_mel_spectrogram(audio).to(model_m.device) #Compute the mel spectrogram
_, probs = model_m.detect_language(mel) #Detect the spoken language
print("Detected language: " + list(probs.keys())[0])