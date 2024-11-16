import os
import json
import openai
import nltk
import re

# Initialize NLTK resources
nltk.download("punkt")

openai.api_key = "pk-CigHzsmOuWnIaohAxYhWfOjhuXTVOUdEQJBqmSDCWxIHjuiB"
openai.base_url = "https://api.pawan.krd/pai-001/v1/"

TXT_FILES_DIR = r"C:\Users\krish\OneDrive-MSFT\Subjects5thSemester\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"
KNOWLEDGE_BASE_PATH = r"C:\Users\krish\OneDrive-MSFT\Subjects5thSemester\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\knowledge_base.json"


def load_knowledge_base():
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_knowledge_base(knowledge_base):
    with open(KNOWLEDGE_BASE_PATH, "w") as f:
        json.dump(knowledge_base, f, indent=4)


def generate_questions_answers(content):
    truncated_content = content[:1500] if len(content) > 1500 else content

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate exactly 5 question-answer pairs based on the following text. "
                    f"Answers should be concise and clear, with some questions involving numeric details.\n\n"
                    f"Format strictly as:\n"
                    f"Q1: <question>\nA1: <answer>\nQ2: <question>\nA2: <answer>\n"
                    f"Text:\n{truncated_content}"
                ),
            }
        ],
        max_tokens=500,
        temperature=0.5,
    )

    generated_text = response.choices[0].message.content
    print(f"Generated raw Q&A text for debugging:\n{generated_text}\n")

    questions_answers = []

    # Split the text by questions "Q1:", "Q2:", etc.
    questions = re.split(r"Q\d+:", generated_text)
    for i in range(1, len(questions)):
        question_part = questions[i].strip()

        # Look for the answer immediately following the question
        answer_match = re.search(r"A\d+:", question_part)
        if answer_match:
            question = question_part[: answer_match.start()].strip()
            answer = question_part[answer_match.end() :].strip()

            questions_answers.append({"question": question, "answer": answer})

    return questions_answers


def process_text_files(knowledge_base):
    for filename in os.listdir(TXT_FILES_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(TXT_FILES_DIR, filename)
            with open(file_path, "r") as f:
                content = f.read()

            questions_answers = generate_questions_answers(content)

            if len(questions_answers) == 5:
                knowledge_base[filename] = questions_answers
                print(f"Processed '{filename}' with 5 Q&A pairs.")
            else:
                print(
                    f"Warning: '{filename}' generated {len(questions_answers)} Q&A pairs; skipping this file."
                )


def main():
    knowledge_base = load_knowledge_base()
    process_text_files(knowledge_base)
    save_knowledge_base(knowledge_base)
    print("Knowledge base updated successfully.")


if __name__ == "__main__":
    main()
