# import gradio as gr
# import os
# import torch
# import whisper
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer, util

# # Set the device to GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Load models
# model_m = whisper.load_model("medium", device=device)
# qa_pipeline = pipeline(
#     'question-answering',
#     model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
#     device=0 if device == "cuda" else -1
# )
# semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Helper functions
# def transcribe_question(input_question_file):
#     question_audio, sr = librosa.load(input_question_file, sr=16000)
#     whisper_audio = torch.tensor(question_audio, dtype=torch.float32).to(device)
#     result = model_m.transcribe(whisper_audio, language="en", fp16=torch.cuda.is_available())
#     return result["text"]

# def generate_answer(question_text, context):
#     answer = qa_pipeline({
#         "question": question_text,
#         "context": context
#     })
#     return answer["answer"]

# def semantic_search(query, segments):
#     query_embedding = semantic_model.encode(query, convert_to_tensor=True)
#     segment_embeddings = semantic_model.encode(segments, convert_to_tensor=True)
#     similarities = util.pytorch_cos_sim(query_embedding, segment_embeddings)[0]
#     most_similar_idx = torch.argmax(similarities).item()
#     return segments[most_similar_idx]

# def speech_based_qa_pipeline(input_question_file, transcription_files):
#     question_text = transcribe_question(input_question_file)
#     segments = []
#     for file_path in transcription_files:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             segments.append(f.read().strip())
#     relevant_segment = semantic_search(question_text, segments)
#     final_answer = generate_answer(question_text, relevant_segment)
#     return f"Question: {question_text}\nAnswer: {final_answer}"

# # Dummy function for Gradio interface
# def dummy_function(audio_file):
#     # Path to transcription files
#     transcription_files = [
#         r"/content/drive/MyDrive/refined_transription.txt"
#         r"/content/drive/MyDrive/combined_output1.txt"
#         # Add more transcription files here
#     ]
    
#     # Call the pipeline with audio file and transcription files
#     answer = speech_based_qa_pipeline(audio_file, transcription_files)
#     return answer

# # Set up Gradio interface
# iface = gr.Interface(
#     fn=dummy_function,
#     inputs=gr.Audio(type="filepath"),  # Audio file input
#     outputs="text",                    # Output as text
#     title="Speech-Based QA System",
#     description="Upload an audio file containing your question, and the system will process the question and return an answer."
# )

# iface.launch(debug=True)

import gradio as gr
import os
import torch
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

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
    
    # Store the answers from each transcription file
    answers = []
    
    for file_path in transcription_files:
        segments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            segments.append(f.read().strip())
        
        relevant_segment = semantic_search(question_text, segments)
        final_answer = generate_answer(question_text, relevant_segment)
        answers.append(final_answer)
    
    # Compare answers (you can implement a method to select the best answer based on confidence, similarity, etc.)
    best_answer = max(answers, key=len)  # Just an example: selecting the longest answer (you can change the comparison logic)
    
    return f"Question: {question_text}\nBest Answer: {best_answer}"

# Dummy function for Gradio interface
def dummy_function(audio_file):
    # Path to transcription files
    transcription_files = [
        r"/content/drive/MyDrive/refined_transription.txt",
        r"/content/drive/MyDrive/combined_output1.txt"
        # Add more transcription files here
    ]
    
    # Call the pipeline with audio file and transcription files
    answer = speech_based_qa_pipeline(audio_file, transcription_files)
    return answer

# Set up Gradio interface
iface = gr.Interface(
    fn=dummy_function,
    inputs=gr.Audio(type="filepath"),  # Audio file input
    outputs="text",                    # Output as text
    title="Speech-Based QA System",
    description="Upload an audio file containing your question, and the system will process the question and return the best answer from multiple transcription files."
)

iface.launch(debug=True)
