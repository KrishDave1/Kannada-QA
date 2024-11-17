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

# Database and collection setup for feedback
feedback_db = client["Feedback_DB"]
feedback_collection = feedback_db["feedback"]

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
    retriever = vectorStore.as_retriever(search_kwargs={"K": 61})
    retrieved_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    response = qa_pipeline(question=query, context=context)
    return response['answer']

def handle_query_with_feedback(audio, feedback_text=None):
    """
    Handles user queries from audio input, processes, returns results, and stores feedback.
    """
    # Transcribe the audio input
    query = transcribe_audio(audio)
    answer = query_data(query)

    # Store the feedback if provided
    if feedback_text:
        feedback_collection.insert_one({"query": query, "answer": answer, "feedback": feedback_text})
        print("Feedback stored successfully!")

    return f"Question: {query}\nAnswer: {answer}"

# Create Gradio interface with feedback input
interface = gr.Interface(
    fn=handle_query_with_feedback,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Textbox(placeholder="Provide feedback on the answer here (optional)")
    ],
    outputs="text",
    title="Voice-based QA System with Feedback",
    description="Ask questions using your voice in Kannada or English, get answers, and provide feedback."
)

# Launch the interface
interface.launch(debug=True)



