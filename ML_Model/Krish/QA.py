import os
from transformers import pipeline


# Step 1: Load and Combine Text Files
def load_text_files(directory_path):
    # Append all text from files in the directory
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


# Step 3: Function to Find Best Answer Across Files
def get_best_answer(question, contexts):
    best_answer = ""
    best_score = 0

    for context in contexts:
        # Updated to pass the question and context as keyword arguments
        result = qa_pipeline(question=question, context=context)
        if result["score"] > best_score:
            best_answer = result["answer"]
            best_score = result["score"]

    return best_answer


# Step 4: Main Function to Run QA
def main():
    directory_path = r"c:\Users\krish\OneDrive-MSFT\Subjects5thSemester\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"
    data = load_text_files(directory_path)

    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = get_best_answer(question, data)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
