# Voice-based QA System with Feedback

## Overview

This project is a voice-based question-answering system that accepts audio inputs in Kannada or English, processes them for answers, and supports user feedback for improved responses. Users can provide corrections to answers, and these are stored for future reference.

## Features

- **Audio Input**: Accepts audio queries for processing.
- **Whisper Model Integration**: Transcribes audio input to text using OpenAI's Whisper.
- **Vector Store with MongoDB**: Integrates MongoDB Atlas for storing and retrieving vectorized documents.
- **Question-Answering Pipeline**: Uses a transformer model for generating answers based on retrieved document context.
- **Feedback Mechanism**: Supports user feedback for answer corrections and updates.
- **Interactive UI**: Built with Gradio for easy user interaction.

## Prerequisites

- Python 3.8+
- MongoDB Atlas setup
- Hugging Face account and API token
- CUDA-compatible GPU (optional for improved performance)

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