import os
import openai

# Set OpenAI API credentials
openai.api_key = "pk-CigHzsmOuWnIaohAxYhWfOjhuXTVOUdEQJBqmSDCWxIHjuiB"
openai.base_url = "https://api.pawan.krd/pai-001/v1/"

# Directories for input and output
input_dir = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\output\translations"  # Directory containing the original text files
output_dir = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\ML_Model\Krish\refined_data"  # Directory to store refined summary files

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


# Define summarization function
def summarize_text(content):
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Please summarize this into a paragraph. It should contain all the important details of the paragraph, including any important statistics and numbers.Here is the below paragraph:\n\n{content}",
            },
        ],
    )

    return completion.choices[0].message.content

# Loop through all text files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_dir, filename)

        # Read the content of the current text file
        with open(input_file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Summarize the content
        summarized_content = summarize_text(content)

        # Write the summarized content to a new file in the output directory
        output_file_path = os.path.join(output_dir, filename)
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(summarized_content)

        print(f"Summarized {filename} and saved to {output_file_path}")

print("All files have been summarized and saved in the 'refined_data' directory.")
