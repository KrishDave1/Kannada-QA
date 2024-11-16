import os
import torch
import whisper
import librosa
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
model_m = whisper.load_model("medium", device=device)
qa_pipeline = pipeline(
    'question-answering',
    model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    device=0 if device == "cuda" else -1
)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions
def transcribe_question(input_question_file):
    question_audio, sr = librosa.load(input_question_file, sr=16000)
    whisper_audio = torch.tensor(question_audio, dtype=torch.float32).to(device)
    result = model_m.transcribe(whisper_audio, language="en", fp16=torch.cuda.is_available())
    return result["text"]

def generate_answer(question_text, context):
    answer = qa_pipeline({
        "question": question_text,
        "context": context
    })
    return answer["answer"]

def semantic_search(query, segments):
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    segment_embeddings = semantic_model.encode(segments, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, segment_embeddings)[0]
    most_similar_idx = torch.argmax(similarities).item()
    return segments[most_similar_idx]

def speech_based_qa_pipeline(input_question_file, transcription_files):
    question_text = transcribe_question(input_question_file)
    segments = []
    for file_path in transcription_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            segments.append(f.read().strip())
    relevant_segment = semantic_search(question_text, segments)
    final_answer = generate_answer(question_text, relevant_segment)
    return question_text, final_answer

# Gradio function wrapper
def gradio_pipeline(audio_file):
    if isinstance(audio_file, str):
        file_path = audio_file
    else:
        # Save the file buffer to a temporary file
        temp_file_path = "temp_audio_file.wav"
        with open(temp_file_path, "wb") as f:
            f.write(audio_file.read())
        file_path = temp_file_path

    # Path to transcription files
    transcription_files = [
        r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\combined_output1.txt"
    ]

    question_text, final_answer = speech_based_qa_pipeline(file_path, transcription_files)
    
    # Clean up temporary file if created
    if file_path == "temp_audio_file.wav":
        os.remove(file_path)

    return f"Question: {question_text}\nAnswer: {final_answer}"

# Gradio Interface
iface = gr.Interface(
    fn=gradio_pipeline,
    inputs=gr.Audio(type="filepath"),  # Use 'filepath' for audio
    outputs="text",
    title="Speech-Based QA System",
    description="Upload an audio file containing your question. The system will process the question and return an answer based on the knowledge base."
)

iface.launch(debug=True)
