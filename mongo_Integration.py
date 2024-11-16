from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests
import openai
from urllib.parse import quote_plus
from pymongo import MongoClient
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
# from langchain.llms import openai
# from langchain.chains import retrieval_qa
from langchain_community.llms import openai
from langchain_community import retrievers
import gradio
from gradio.themes.base import Base
import os

print(os.getenv("OPEN_API_KEY"))

# Encode the username and password
username = quote_plus("valmik0000000")
password = quote_plus("valmik@mongo7")  # Replace with your actual password

uri = f"mongodb+srv://{username}:{password}@valmikcluster0.hdqee.mongodb.net/?retryWrites=true&w=majority&appName=ValmikCluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

hf_token = "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"

dbName = "ML_Fiesta"
collectionName = "translations"
collection = client[dbName][collectionName]

hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

open_api_key = "pk-CigHzsmOuWnIaohAxYhWfOjhuXTVOUdEQJBqmSDCWxIHjuiB"

loader = DirectoryLoader(r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data", glob="./*.txt", show_progress=True)
data = loader.load()

# embeddings = OpenAIEmbeddings(open_api_key="pk-CigHzsmOuWnIaohAxYhWfOjhuXTVOUdEQJBqmSDCWxIHjuiB")

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY") or "pk-CigHzsmOuWnIaohAxYhWfOjhuXTVOUdEQJBqmSDCWxIHjuiB")

vectorStore = MongoDBAtlasVectorSearch.from_documents(data, embeddings, collection=collection)