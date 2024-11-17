# Automatic Speech Recognition and Question-Answering Model

### NOTE : The final file is located in /ML_Model/Ml-Fiesta.ipynb. Just run this file to get the application.Other files are all the other attempts we made while making the ML Model

This repository demonstrates an integrated system for **Automatic Speech Recognition (ASR)** and **Contextual Question-Answering (QA)**. Combining cutting-edge tools like OpenAI Whisper, Hugging Face Transformers, and MongoDB Atlas Vector Search, the system provides seamless processing of audio inputs, multilingual transcription and translation, and contextual question answering with feedback mechanisms.

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Usage Workflow](#usage-workflow)  
4. [System Components](#system-components)  
   - [Automatic Speech Recognition and Translation](#automatic-speech-recognition-and-translation)  
   - [Question-Answering Model](#question-answering-model)  
   - [Feedback Mechanism](#feedback-mechanism)  
5. [Challenges and Solutions](#challenges-and-solutions)  
6. [Potential Enhancements](#potential-enhancements)  
7. [Conclusion](#conclusion)

---

## Prerequisites

- Python 3.8+
- MongoDB Atlas setup
- Hugging Face account and API token
- CUDA-compatible GPU (optional for improved performance)

## Overview

This project integrates the following:

- **ASR and Translation:** Converts audio inputs into transcriptions and English translations using the Whisper model.
- **QA System:** Answers user queries based on contextual document retrieval using MongoDB Atlas Vector Search and a fine-tuned BERT model.
- **Feedback Mechanism:** Collects and processes user feedback to improve system performance.

---

## Key Features
- **Audio Input**: Accepts audio queries for processing.
- **Whisper Model Integration**: Transcribes audio input to text using OpenAI's Whisper.
- **Vector Store with MongoDB**: Integrates MongoDB Atlas for storing and retrieving vectorized documents.
- **Question-Answering Pipeline**: Uses a transformer model for generating answers based on retrieved document context.
- **Feedback Mechanism**: Supports user feedback for answer corrections and updates.
- **Interactive UI**: Built with Gradio for easy user interaction.

---

## Usage Workflow

1. **Setup:**
   - Install dependencies (e.g., `whisper`, `librosa`, `transformers`, etc.).
   - Configure MongoDB Atlas credentials.

2. **Audio Processing:**
   - Provide an audio file path for processing.
   - Obtain transcriptions and English translations in organized directories.

3. **Question Answering:**
   - Store textual documents in the specified MongoDB directory.
   - Use the QA pipeline to retrieve and answer questions based on the stored context.

4. **Feedback Collection:**
   - Submit voice queries and feedback through the Gradio interface.
   - Refine answers and persist interactions in a JSON file for analysis.

---

## System Components

### Automatic Speech Recognition and Translation

- **Language Detection:** Identifies the language of the input audio.
- **Transcription and Translation:** Generates transcripts in the original language and English translations.
- **Chunked Processing:** Splits audio files for memory-efficient handling.

### Question-Answering Model

- **Embedding Model:** Converts text into vector representations for semantic search.
- **Vector Store:** Manages document embeddings in MongoDB for efficient retrieval.
- **QA Pipeline:** Uses a pre-trained BERT model fine-tuned on the SQuAD dataset.

### Feedback Mechanism

- **Interactive Interface:** Built with Gradio for real-time query submission and feedback.
- **Persistent Storage:** Stores feedback in JSON format for system refinement.

---

## Challenges and Solutions

- **Large Dataset Handling:** Utilizes MongoDB vector indexing for efficient retrieval.  
- **Accent and Noise Variability:** Pre-trained Whisper model improves accuracy for diverse inputs.  
- **Memory Efficiency:** Processes audio in smaller chunks to prevent crashes.  

---

## Potential Enhancements

- **Real-Time Processing:** Add streaming capabilities for live transcription and QA.  
- **Advanced QA Models:** Explore GPT-based models for improved performance.  
- **Noise Reduction:** Integrate preprocessing to enhance transcription accuracy.  
- **User Interface:** Develop a web-based front end for better accessibility.  

---

## Conclusion

This system showcases a robust integration of ASR, semantic search, and QA, suitable for diverse applications like multilingual transcription, contextual search, and interactive voice systems. Its scalable design and feedback-driven approach make it a powerful tool for real-world deployments.

---

Feel free to use and modify this README template to suit your repository's specific requirements!

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/valmikGit/ML-Fiesta-Byte-Synergy-Hackathon
   ```
2. **Create a Python virtual environment**:
    ```bash
    python -m venv projectenv
    ```
3. **Activate the virtual environment**:
    ```bash
    projectenv\Scripts\activate
    ```
4. **Install the necessary requirements**:
    ```bash
    pip install -r final_requirements.txt
    ```
