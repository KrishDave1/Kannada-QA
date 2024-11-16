
import gradio as gr
import os
import torch
import librosa
import whisper
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import json

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

# Path to feedback file
feedback_file = "feedback.json"

# Load feedback data
if os.path.exists(feedback_file):
    with open(feedback_file, "r", encoding="utf-8") as f:
        feedback_data = json.load(f)
else:
    feedback_data = {}

# Helper functions
def save_feedback(question, correct_answer):
    """Save user feedback to the feedback file."""
    feedback_data[question] = correct_answer
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=4)

def transcribe_question(input_question_file):
    question_audio, sr = librosa.load(input_question_file, sr=16000)
    whisper_audio = torch.tensor(question_audio, dtype=torch.float32).to(device)
    result = model_m.transcribe(whisper_audio, language="en", fp16=torch.cuda.is_available())
    return result["text"]

def generate_answer(question_text, context):
    """Generate an answer using the QA pipeline."""
    answer = qa_pipeline({
        "question": question_text,
        "context": context
    })
    return answer["answer"], answer["score"]

def semantic_search(query, segments):
    """Search for the most relevant segment using semantic similarity."""
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    segment_embeddings = semantic_model.encode(segments, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, segment_embeddings)[0]
    most_similar_idx = torch.argmax(similarities).item()
    return segments[most_similar_idx]

def fetch_best_answer(question_text, answers_with_scores):
    """
    Fetch the best answer based on scores and semantic similarity.
    This combines both answer relevance and context match.
    """
    # Sort answers based on their score
    sorted_answers = sorted(answers_with_scores, key=lambda x: x[1], reverse=True)
    best_answer, best_score = sorted_answers[0]
    return best_answer

def speech_based_qa_pipeline(input_question_file, transcription_files):
    question_text = transcribe_question(input_question_file)

    # Check if the question exists in feedback
    if question_text in feedback_data:
        return f"Question: {question_text}\nAnswer (from feedback): {feedback_data[question_text]}"

    # Process the question as usual
    answers_with_scores = []

    for file_path in transcription_files:
        segments = []
        with open(file_path, 'r', encoding='utf-8') as f:
            segments.append(f.read().strip())

        relevant_segment = semantic_search(question_text, segments)
        answer, score = generate_answer(question_text, relevant_segment)
        answers_with_scores.append((answer, score))

    # Fetch the best answer using enhanced logic
    best_answer = fetch_best_answer(question_text, answers_with_scores)

    return f"Question: {question_text}\nBest Answer: {best_answer}"

def feedback_handler(audio_file, corrected_answer):
    transcription_files = [
        r"/content/drive/MyDrive/refined_transription.txt",
        r"/content/drive/MyDrive/combined_output1.txt"
        # Add more transcription files here
    ]

    # Get the question text
    question_text = transcribe_question(audio_file)

    # Perform semantic similarity check before saving feedback
    if question_text in feedback_data:
        previous_answer = feedback_data[question_text]
        similarity_score = util.pytorch_cos_sim(
            semantic_model.encode(corrected_answer, convert_to_tensor=True),
            semantic_model.encode(previous_answer, convert_to_tensor=True)
        ).item()
        if similarity_score < 0.8:
            save_feedback(question_text, corrected_answer)
            return f"Thank you! Feedback recorded and updated for question: {question_text}"
        else:
            return f"The provided answer is similar to the existing one. No changes made."
    else:
        save_feedback(question_text, corrected_answer)
        return f"Thank you! Feedback recorded for question: {question_text}"

# Gradio interface with submit button
with gr.Blocks(css=".orange-button {background-color: orange; color: white; font-weight: bold;}") as app:
    gr.Markdown("# Speech-Based QA System with Feedback")
    gr.Markdown("Upload an audio file containing your question, and the system will return the best answer. If the answer is incorrect, you can provide feedback.")

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload your question (audio)")

    submit_button = gr.Button("Submit", elem_classes="orange-button")  # Orange button below the audio input
    output_text = gr.Textbox(label="Answer", lines=3)

    feedback_input = gr.Textbox(label="Provide correct answer if incorrect")
    feedback_button = gr.Button("Submit Feedback")

    def main_function(audio_file):
        transcription_files = [
            r"/content/drive/MyDrive/refined_transription.txt",
            r"/content/drive/MyDrive/combined_output1.txt"
            # Add more transcription files here
        ]
        return speech_based_qa_pipeline(audio_file, transcription_files)

    # Link the submit button to the main function
    submit_button.click(main_function, inputs=audio_input, outputs=output_text)

    # Feedback submission
    feedback_button.click(feedback_handler, inputs=[audio_input, feedback_input], outputs=output_text)

app.launch(debug=True)
