import os
from transformers import pipeline

# Step 1: Load and Combine Text Files
def load_text_files(directory_path):
    data = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            if file_content.strip():  # Only add non-empty files
                data.append(file_content)
    print(f"Loaded {len(data)} documents.")  # Check number of loaded documents
    return data


# Step 2: Initialize the QA Model
qa_pipeline = pipeline(
    "question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"
)


# Step 3: Function to Find Best Answer Across Files
def get_best_answer(question, contexts):
    best_answer = ""
    best_score = 0

    for context in contexts:
        print(f"Context preview: {context[:100]}...")  # Print the first 100 characters of context
        result = qa_pipeline(question=question, context=context)
        print(f"Score: {result['score']}, Answer: {result['answer']}")  # Debug the result
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer if best_answer else "Sorry, I couldn't find an answer."


# Step 4: Main Function to Run QA
def main():
    directory_path = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\output\translations"
    data = load_text_files(directory_path)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = get_best_answer(question, data)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
