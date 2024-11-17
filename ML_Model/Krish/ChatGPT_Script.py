import os
import re
import openai

# Set OpenAI API credentials
openai.api_key = "pk-CigHzsmOuWnIaohAxYhWfOjhuXTVOUdEQJBqmSDCWxIHjuiB"
openai.base_url = "https://api.pawan.krd/pai-001/v1/"

# Directories for input and output
input_dir = r"C:\Users\krish\OneDrive-MSFT\Subjects5thSemester\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\output"
output_dir = r"C:\Users\krish\OneDrive-MSFT\Subjects5thSemester\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Define preprocessing function to clean input text
def clean_text(content):
    # Remove HTML tags
    content = re.sub(r"<[^>]+>", "", content)
    # Remove extra whitespace and special characters
    content = re.sub(r"\s+", " ", content).strip()
    return content


# Define summarization function
def refine_text(content):
    prompt = (
        "You are refining text for a BERT-based QA model. Please remove redundant, irrelevant, or non-informative text while retaining all "
        "important details, facts, and statistics from the input. The output must be clear, concise, and comprehensive, focusing on "
        "providing the full context for the QA model. Here is the input:\n\n"
        f"{content}"
    )

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.5,
    )

    return completion.choices[0].message.content


# Process all text files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_dir, filename)

        # Read and clean the content of the current text file
        with open(input_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            cleaned_content = clean_text(content)

        # Refine the content
        refined_content = refine_text(cleaned_content)

        # Postprocess output to remove any unwanted artifacts
        refined_content = clean_text(refined_content)

        # Write the refined content to a new file in the output directory
        output_file_path = os.path.join(output_dir, filename)
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(refined_content)

        print(f"Refined {filename} and saved to {output_file_path}")

print("All files have been refined and saved in the 'refined_data' directory.")
