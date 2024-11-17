import gradio as gr
from transformers import pipeline
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader

# MongoDB setup
username = quote_plus("valmik0000000")
password = quote_plus("valmik@mongo7")  # Replace with your actual password

uri = f"mongodb+srv://{username}:{password}@valmikcluster0.hdqee.mongodb.net/?retryWrites=true&w=majority&appName=ValmikCluster0"
client = MongoClient(uri, server_api=ServerApi('1'), tls=True)

dbName = "ML_Fiesta"
collectionName = "translations"
collection = client[dbName][collectionName]

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

if collection.estimated_document_count() == 0:
    print("No data found in the vector store. Loading documents and creating the vector store.")
    loader = DirectoryLoader(
        r"./refined_data",  # Update to your file directory
        glob="./*.txt",
        show_progress=True
    )
    data = loader.load()
    vectorStore = MongoDBAtlasVectorSearch.from_documents(data, collection=collection, embedding=embeddings)
else:
    vectorStore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings)

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad", device=0)
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")

def query_data(query):
    retriever = vectorStore.as_retriever(search_kwargs={"K": 5})
    retrieved_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    response = qa_pipeline(question=query, context=context)
    return response['answer'], context

def process_audio(file, query):
    if file:
        transcription = asr_pipeline(file)["text"]
        question = transcription if not query else query
    else:
        question = query
    
    answer, context = query_data(question)
    return f"Question: {question}\n\nAnswer: {answer}\n\nContext: {context}"

def feedback_handler(question, answer, feedback):
    if feedback:
        collection.insert_one({"question": question, "answer": answer, "feedback": feedback})
        return "Feedback saved. Thank you!"
    return "No feedback provided."

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# Audio Q&A with Feedback")
    
    with gr.Row():
        audio_input = gr.Audio(label="Record or Upload Audio", type="filepath")
        query_input = gr.Textbox(label="Type a Question (Optional)")
    
    answer_output = gr.Textbox(label="Answer")
    feedback_input = gr.Textbox(label="Provide Feedback (Optional)")
    feedback_status = gr.Textbox(label="Feedback Status", interactive=False)
    
    with gr.Row():
        submit_button = gr.Button("Submit")
        feedback_button = gr.Button("Submit Feedback")
    
    submit_button.click(process_audio, inputs=[audio_input, query_input], outputs=[answer_output])
    feedback_button.click(feedback_handler, inputs=[query_input, answer_output, feedback_input], outputs=[feedback_status])

# Run the app
app.launch()
