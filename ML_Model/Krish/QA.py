import os
import json
from transformers import pipeline


# Step 1: Load and Combine Text Files
def load_text_files(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            data.append(file.read())
    return data


# Step 2: Initialize the QA Model
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)


# Step 3: Load Knowledge Base
def load_knowledge_base():
    try:
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# Step 4: Save Knowledge Base
def save_knowledge_base(knowledge_base):
    with open("knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=4)


# Step 5: Function to Find Best Answer Across Files
def get_best_answer(question, contexts, knowledge_base):
    # Check if an updated answer is already available in the knowledge base
    if question in knowledge_base:
        return knowledge_base[question]

    best_answer = ""
    best_score = 0
    for context in contexts:
        result = qa_pipeline(question=question, context=context)
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer


# Step 6: Main Function with Feedback Mechanism
def main():
    directory_path = r"C:\Users\krish\OneDrive-MSFT\Subjects5thSemester\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"
    data = load_text_files(directory_path)
    knowledge_base = load_knowledge_base()

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = get_best_answer(question, data, knowledge_base)
        print(f"Answer: {answer}")

        # Get feedback from the user
        feedback = input("Was this answer correct? (yes/no): ").strip().lower()
        if feedback == "no":
            correct_answer = input("Please provide the correct answer: ")
            knowledge_base[question] = correct_answer
            save_knowledge_base(knowledge_base)
            print("Thank you! The answer has been updated.")

        print("\n")


if __name__ == "__main__":
    main()
