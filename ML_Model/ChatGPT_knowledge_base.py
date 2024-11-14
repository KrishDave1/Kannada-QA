import os
import json
import openai  # Uncomment if using OpenAI API for question-answer generation
import nltk
from nltk.tokenize import sent_tokenize

# Initialize NLTK resources
nltk.download('punkt')

# Set OpenAI API key (if using OpenAI API)
openai.api_key = "pk-TZfTgxpJUKyFRTsfNFnoQjxZxXrppMbbusgmIfqFNsTUzTFW"
openai.base_url = "https://api.pawan.krd/pai-001/v1/"

# Directory containing .txt files
TXT_FILES_DIR = r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"
KNOWLEDGE_BASE_PATH = r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\knowledge_base.json"

def load_knowledge_base():
    """Load or create a knowledge base JSON file."""
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_knowledge_base(knowledge_base):
    """Save updated knowledge base to JSON file."""
    with open(KNOWLEDGE_BASE_PATH, 'w') as f:
        json.dump(knowledge_base, f, indent=4)

def generate_questions_answers(content):
    """Generate 5 questions and answers based on text content using ChatCompletion API."""
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",  # or use "gpt-4" if you have access
    #     messages=[
    #         {"role": "system", "content": "Generate 5 question-answer pairs based on the following text."},
    #         {"role": "user", "content": content}
    #     ],
    #     max_tokens=500,
    #     temperature=0.5,
    # )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or use "gpt-4" if you have access
        messages=[
            {
                "role": "user",
                "content": f"Generate 5 question-answer pairs based on the following text:\n\n{content}"
            }
        ],
        max_tokens=500,
        temperature=0.5,
    )
    
    generated_text = response.choices[0].message['content'].strip()
    pairs = [tuple(qa.split("A:")) for qa in generated_text.split("Q:") if "A:" in qa]
    
    questions_answers = {}
    for question, answer in pairs:
        questions_answers[question.strip()] = answer.strip()
        
    return questions_answers

def process_text_files(knowledge_base):
    """Process each .txt file in the folder to generate Q&A pairs."""
    for filename in os.listdir(TXT_FILES_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(TXT_FILES_DIR, filename)
            with open(file_path, 'r') as f:
                content = f.read()

            # Generate questions and answers for the current file
            questions_answers = generate_questions_answers(content)

            # Update knowledge base with new questions and answers
            knowledge_base.update(questions_answers)

def main():
    # Load or initialize the knowledge base
    knowledge_base = load_knowledge_base()
    
    # Process each .txt file to generate and add Q&A pairs
    process_text_files(knowledge_base)
    
    # Save the updated knowledge base
    save_knowledge_base(knowledge_base)
    print("Knowledge base updated successfully.")

if __name__ == "__main__":
    main()