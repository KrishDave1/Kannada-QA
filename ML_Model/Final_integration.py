import gradio as gr
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from transformers import pipeline
import whisper
import torch
import json
import os

# MongoDB setup
username = quote_plus("valmik0000000")
password = quote_plus("valmik@mongo7")  # Replace with your actual password

uri = f"mongodb+srv://{username}:{password}@valmikcluster0.hdqee.mongodb.net/?retryWrites=true&w=majority&appName=ValmikCluster0"
client = MongoClient(uri, server_api=ServerApi('1'), tls=True)

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Error: {e}")

# Feedback file
FEEDBACK_FILE = "feedback.json"

# Load or initialize feedback data
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r") as f:
        feedback_data = json.load(f)
else:
    feedback_data = {}

# Save feedback to file
def save_feedback_to_file():
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback_data, f, indent=4)

# Database and collection setup for QA
dbName = "ML_Fiesta"
collectionName = "translations"
collection = client[dbName][collectionName]

# Define the embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Check if the data already exists in the vector store
if collection.estimated_document_count() == 0:
    print("No data found in the vector store. Loading documents and creating the vector store.")
    loader = DirectoryLoader(
        r"/content/drive/MyDrive/ML_Fiesta_Mongo_DB_Vectorization/translations",
        glob="./*.txt",
        show_progress=True
    )
    data = loader.load()
    vectorStore = MongoDBAtlasVectorSearch.from_documents(data, collection=collection, embedding=embeddings)
else:
    print("Data already exists in the vector store. Connecting to existing vector store.")
    vectorStore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings)

# Define the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad", device=0)

# Load Whisper model for audio transcription
whisper_model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

# Global variables to store the last query and answer
last_query = ""
last_answer = ""

def transcribe_audio(audio_file):
    """
    Transcribes the given audio file using Whisper.
    """
    result = whisper_model.transcribe(audio_file)
    return result['text']

def query_data(query) -> str:
    """
    Querying data loaded in MongoDB and returning an answer.
    """
    # Check for feedback corrections in the feedback file
    if query in feedback_data:
        print("Using corrected answer from feedback file.")
        return feedback_data[query]  # Use the corrected answer

    # If no corrections, use the vector store and QA pipeline
    retriever = vectorStore.as_retriever(search_kwargs={"K": 61})
    retrieved_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    response = qa_pipeline(question=query, context=context)
    return response['answer']

def handle_query(audio):
    """
    Handles user queries from audio input and returns results.
    """
    global last_query, last_answer
    # Transcribe the audio input
    query = transcribe_audio(audio)
    last_query = query  # Store the query for later feedback
    answer = query_data(query)
    last_answer = answer  # Store the answer for later correction

    # Return the question and the answer
    return f"Question: {query}\nAnswer: {answer}"

# def submit_feedback(feedback_text):
#     """
#     Handles feedback submission and updates the answer if the feedback indicates a wrong answer.
#     """
#     global last_query, last_answer
#     if last_query == "":
#         return "No query has been submitted yet. Please submit a query first."

#     if feedback_text.lower() in ['wrong', 'incorrect', 'not correct']:
#         # Prompt the user for the correct answer
#         corrected_answer = input(f"Please provide the correct answer for the question '{last_query}': ")
#         # Store the feedback and corrected answer in the feedback file
#         feedback_data[last_query] = corrected_answer
#         save_feedback_to_file()
#         print("Feedback stored successfully! Answer has been corrected.")
#         return f"The answer has been corrected. The new answer is: {corrected_answer}"
#     else:
#         # Store feedback without corrections
#         feedback_data[last_query] = last_answer
#         save_feedback_to_file()
#         print("Feedback stored successfully!")
#         return f"Feedback for the query '{last_query}' has been stored."
def submit_feedback(feedback_text):
    """
    Handles feedback submission and updates the answer if the feedback indicates a wrong answer.
    """
    global last_query, last_answer
    if last_query == "":
        return "No query has been submitted yet. Please submit a query first."

    if feedback_text.lower() in ['wrong', 'incorrect', 'not correct']:
        # Prompt the user for the correct answer
        corrected_answer = input(f"Please provide the correct answer for the question '{last_query}': ")
        if corrected_answer.strip():  # Ensure the corrected answer is not empty
            # Store the feedback and corrected answer in the feedback file
            feedback_data[last_query] = corrected_answer
            save_feedback_to_file()
            print("Feedback stored successfully! Answer has been corrected.")
            return f"The answer has been corrected. The new answer is: {corrected_answer}"
        else:
            return "No corrected answer provided. Feedback not stored."
    else:
        # Store feedback without corrections
        feedback_data[last_query] = last_answer
        save_feedback_to_file()
        print("Feedback stored successfully!")
        return f"Feedback for the query '{last_query}' has been stored."


# Gradio Blocks Setup
with gr.Blocks() as query_block:
    gr.Markdown("# Voice-based QA System with Feedback")
    gr.Markdown("Ask questions using your voice in Kannada or English, get answers, and provide feedback. If an answer is incorrect, you can provide the correct answer, and it will be stored.")

    with gr.Row():
        with gr.Column():
            query_input = gr.Audio(type="filepath", label="Upload your audio query")
            query_button = gr.Button("Submit Query")
            query_output = gr.Textbox(label="Response")
        
        with gr.Column():
            feedback_input = gr.Textbox(placeholder="Provide feedback on the answer here", label="Feedback")
            feedback_button = gr.Button("Submit Feedback")
            feedback_output = gr.Textbox(label="Feedback Result")
    
    query_button.click(fn=handle_query, inputs=query_input, outputs=query_output)
    feedback_button.click(fn=submit_feedback, inputs=feedback_input, outputs=feedback_output)

# Launch the Gradio interface
query_block.launch(debug=True)