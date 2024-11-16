import os

# Specify the folder containing the .txt files
input_folder = r"C:\Users\mitta\OneDrive - iiit-b\Documents\ML-Fiesta-Byte-Synergy-Hackathon\output\translations"  # Replace with the folder path
output_file = "combined_output1.txt"  # Output file name

# Open the output file in write mode
with open(output_file, "w", encoding="utf-8") as outfile:
    # Iterate through all files in the folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(input_folder, filename)
            
            # Open each file and append its content to the output file
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    # Skip lines containing headers or footers
                    if line.startswith("--- Start of") or line.startswith("--- End of"):
                        continue
                    outfile.write(line)  # Write content to output file
                outfile.write("\n")  # Add a newline for separation between files

print(f"All files combined into {output_file} without headers or footers.")
