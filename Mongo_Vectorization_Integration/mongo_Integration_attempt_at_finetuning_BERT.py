from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from pymongo import MongoClient
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from transformers import pipeline, BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import os
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
    print(e)

# Database and collection
dbName = "ML_Fiesta"
collectionName = "translations"
collection = client[dbName][collectionName]

# Define the embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Check if the data already exists in the vector store
if collection.estimated_document_count() == 0:
    print("No data found in the vector store. Loading documents and creating the vector store.")
    # Document loader
    loader = DirectoryLoader(
        r"/content/drive/MyDrive/ML_Fiesta_Mongo_DB_Vectorization/refined_data",
        glob="./*.txt",
        show_progress=True
    )
    data = loader.load()

    # Load documents into the vector store
    vectorStore = MongoDBAtlasVectorSearch.from_documents(data, collection=collection, embedding=embeddings)
else:
    print("Data already exists in the vector store. Connecting to existing vector store.")
    vectorStore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings)

# Define the question-answering pipeline with the BERT model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad", device=0)

# Fine-tune the model with your SQuAD-style JSON file
def fine_tune_model(train_file, output_dir="./fine_tuned_model"):
    """
    Fine-tunes the BERT model using a SQuAD-style JSON file.
    """
    # Load the dataset from a SQuAD-style JSON file
    dataset = load_dataset("json", data_files={"train": train_file})

    # Load pre-trained BERT model and tokenizer
    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def preprocess_data(example):
        """
        Preprocesses the data for fine-tuning.
        """
        processed_examples = []
        for paragraph in example['data'][0][0]['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa.get('question', "")
                answers = qa.get('answers', [])
                if answers:
                    answer = answers[0].get('text', "")
                    answer_start = answers[0].get('answer_start', -1)
                    if answer_start != -1:
                        processed_examples.append({
                            'input_ids': tokenizer.encode(question, context, truncation=True, padding='max_length', max_length=512),
                            'labels': tokenizer.encode(answer, add_special_tokens=False)
                        })
        return processed_examples

    # Flatten processed examples and convert to dictionary format
    def process_and_flatten(batch):
        return {key: [item[key] for item in batch if key in item] for key in batch[0]}

    # Preprocess the dataset
    tokenized_data = dataset["train"].map(
        preprocess_data,
        batched=True,
        batch_size=1,
        remove_columns=dataset["train"].column_names
    ).flatten_indices().map(process_and_flatten)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    # Fine-tune the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

# Fine-tune the model with your SQuAD-style JSON file
train_file = "/content/drive/MyDrive/ML_Fiesta_Mongo_DB_Vectorization/refined_data/generated_train_data.json"  # Adjusted path to point to the uploaded file
fine_tune_model(train_file)

# Load the fine-tuned model after training
fine_tuned_model = BertForQuestionAnswering.from_pretrained("./fine_tuned_model")
fine_tuned_tokenizer = BertTokenizer.from_pretrained("./fine_tuned_model")

# Define the query function to use the fine-tuned model
def query_data(query) -> str:
    """
    Querying data loaded in Mongo DB.
    """
    # Use the vector store's retriever to find relevant documents
    retriever = vectorStore.as_retriever(search_kwargs={"K": 61})
    retrieved_docs = retriever.get_relevant_documents(query)

    # Combine retrieved documents as context for the model
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Ask the question with the context
    prompt = f"Context: {context}\n\nQuestion: {query}"

    print(f"Query: {query}")

    # Tokenize the input query and context
    inputs = fine_tuned_tokenizer(query, context, return_tensors="pt", truncation=True, padding=True)

    # Get the answer from the fine-tuned model
    outputs = fine_tuned_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    # Extract the answer
    answer = fine_tuned_tokenizer.convert_tokens_to_string(fine_tuned_tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end+1]))

    print(f"Answer: {answer}")
    return answer

# Query example
queries = [
    "Where have the blood moon thieves been active?", 
    "What is the name of the group involved in illicit activities?", 
    "Which government body has been formed to combat the blood moon thieves?", 
    "Where is the stronghold of the blood moon thieves?",
    "Who does Baba obtain contact information for?",
    "Where does Baba try to meet Somwara Siguna?",
    "What is the potential earnings from one Shrikanda Gita?",
    "How much can be earned from 100 Shrikandas?",
    "When did India's agricultural landscape undergo significant transformation?",
    "Which industries have shown interest in Shrikanda cultivation?",
    "Who is mentioned as an example of successful Shrikanda farming on both small and large scales?",
    "What is the annual demand for Shrikanda?",
    "What does the course offered by the Freedom App Research Team include?"
]

try:
    for query in queries:
        print(query_data(query=query))
except Exception as e:
    print(e)