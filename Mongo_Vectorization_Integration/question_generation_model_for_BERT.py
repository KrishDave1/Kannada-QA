from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer, BertForQuestionAnswering, BertTokenizer
import json
import random
import os
import torch

# Load pre-trained models
question_generator = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")
qa_model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
qa_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to generate questions from the context using T5
def generate_questions(context, num_questions=5):
    inputs = f"generate questions: {context}"
    questions = question_generator(inputs, max_length=100, num_return_sequences=num_questions, do_sample=True)
    return [q['generated_text'] for q in questions]

# Function to answer the generated questions using BERT
def answer_question(context, question):
    # Ensure that the sequence length does not exceed the maximum limit for BERT (512 tokens)
    inputs = qa_tokenizer.encode(question, context, add_special_tokens=True)
    
    # Split context into chunks if the input sequence exceeds the max length
    max_length = 512
    if len(inputs) > max_length:
        inputs = inputs[:max_length]  # Truncate to 512 tokens if the length exceeds
    
    # Convert the inputs to a tensor and add batch dimension
    inputs = torch.tensor(inputs).unsqueeze(0)

    # Make predictions without gradient computation
    with torch.no_grad():
        outputs = qa_model(inputs)

    # Find the start and end positions of the answer
    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax()

    # Extract the answer
    answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs[0][answer_start:answer_end+1]))
    return answer

# Function to split context into chunks of maximum 512 tokens
def split_into_chunks(context, max_length=512):
    tokens = qa_tokenizer.encode(context)
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return chunks

# Function to convert text into SQuAD style
def generate_squad_data(contexts, num_questions=5):
    squad_data = {"data": []}
    for idx, context in enumerate(contexts):
        # Generate questions from the context
        questions = generate_questions(context, num_questions=num_questions)

        # Create Q&A pairs
        for question in questions:
            answers = []
            context_chunks = split_into_chunks(context)
            
            for chunk in context_chunks:
                chunk_context = qa_tokenizer.decode(chunk, skip_special_tokens=True)
                answer = answer_question(chunk_context, question)
                if answer.strip():  # Add non-empty answers
                    answers.append(answer)
            
            # Join answers from chunks (if more than one answer was found)
            combined_answer = " ".join(answers)

            squad_data["data"].append({
                "title": f"document_{idx}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "question": question,
                                "id": f"{idx}_{random.randint(1000,9999)}",  # Random ID for each question
                                "answers": [{"text": combined_answer, "answer_start": context.find(combined_answer)}]
                            }
                        ]
                    }
                ]
            })
    return squad_data

# Example usage
def generate_training_data_from_folder(folder_path, num_questions=5):
    contexts = []
    
    # Loop through all the .txt files in the provided folder path
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                contexts.append(file.read())
    
    # Generate SQuAD-style training data
    squad_data = generate_squad_data(contexts, num_questions=num_questions)

    # Save the result as a JSON file
    train_file_path = os.path.join(folder_path, "generated_train_data.json")
    with open(train_file_path, "w") as f:
        json.dump(squad_data, f, indent=4)

    print(f"Training data saved to {train_file_path}")

# Provide the path to the folder containing .txt files
folder_path = "/content/drive/MyDrive/ML_Fiesta_Mongo_DB_Vectorization/refined_data"  # Update with your actual folder path
generate_training_data_from_folder(folder_path, num_questions=5)