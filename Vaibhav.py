import torch
import torchaudio
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os

# Load processor and model from Hugging Face
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")

# Define the batch processing function
def transcribe_audio_in_batches(file_path, batch_duration=5):
    """
    Transcribes an audio file in batches to avoid memory overflow.
    Args:
    - file_path: Path to the audio file.
    - batch_duration: Duration in seconds for each batch.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Resample if necessary
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Determine the number of samples per batch
    batch_samples = int(batch_duration * 16000)
    num_batches = waveform.shape[1] // batch_samples + int(waveform.shape[1] % batch_samples != 0)
    
    transcript = ""
    for i in range(num_batches):
        # Extract batch
        start = i * batch_samples
        end = start + batch_samples
        batch_waveform = waveform[:, start:end].to("cuda")

        # Process the batch
        inputs = processor(batch_waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to("cuda")).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        batch_transcription = processor.batch_decode(predicted_ids)[0]
        
        transcript += batch_transcription + " "
    
    return transcript.strip()

# Example usage
input_dir = r'C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\audiocorpus'
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        wav_path = os.path.join(input_dir, filename)
        transcript = transcribe_audio_in_batches(wav_path, batch_duration=5)
        print(f"Transcription for {filename}: {transcript}")
