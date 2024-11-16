from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from pymongo import MongoClient
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from transformers import pipeline
import os

# MongoDB setup
username = quote_plus("valmik0000000new")
password = quote_plus("valmik@mongo7")  # Replace with your actual password

uri = f"mongodb+srv://{username}:{password}@valmikcluster0.hdqee.mongodb.net/?retryWrites=true&w=majority&appName=ValmikCluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Database and collection
dbName = "ML_Fiesta"
collectionName = "translations"
collection = client[dbName][collectionName]

# Hugging Face token (optional if required for private models)
hf_token = "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"

# Define the embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Document loader
loader = DirectoryLoader(
    r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data",
    glob="./*.txt",
    show_progress=True
)
data = loader.load()

# Vector store
vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)