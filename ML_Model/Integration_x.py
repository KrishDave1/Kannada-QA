import torch
import whisper
import librosa
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


class SpeechBasedQAModel:
    def __init__(self, asr_model_size="medium", qa_model="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", semantic_model_name='all-MiniLM-L6-v2'):
        # Set device to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper ASR model
        self.asr_model = whisper.load_model(asr_model_size, device=self.device)
        
        # Load QA pipeline
        self.qa_pipeline = pipeline('question-answering', model=qa_model, device=0 if self.device == "cuda" else -1)
        
        # Load Sentence Transformer model
        self.semantic_model = SentenceTransformer(semantic_model_name)

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribes an audio file into text using Whisper ASR.
        """
        question_audio, sr = librosa.load(audio_file, sr=16000)
        whisper_audio = torch.tensor(question_audio, dtype=torch.float32).to(self.device)
        result = self.asr_model.transcribe(whisper_audio, language="en", fp16=torch.cuda.is_available())
        return result["text"]

    def semantic_search(self, query: str, segments: list[str]) -> str:
        """
        Finds the most relevant text segment using semantic search.
        """
        query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
        segment_embeddings = self.semantic_model.encode(segments, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, segment_embeddings)[0]
        most_similar_idx = torch.argmax(similarities).item()
        return segments[most_similar_idx]

    def generate_answer(self, question: str, context: str) -> str:
        """
        Extracts an answer to the question based on the provided context.
        """
        answer = self.qa_pipeline({
            "question": question,
            "context": context
        })
        return answer["answer"]

    def process(self, audio_file: str, context_files: list[str]) -> str:
        """
        End-to-end pipeline for processing the audio question and context files.
        """
        # Transcribe the audio file
        question_text = self.transcribe_audio(audio_file)
        print("Transcribed Question:", question_text)
        
        # Load and concatenate contexts
        segments = []
        for file_path in context_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                segments.append(f.read().strip())
        
        # Find the most relevant context using semantic search
        relevant_segment = self.semantic_search(question_text, segments)
        
        # Generate the final answer
        final_answer = self.generate_answer(question_text, relevant_segment)
        print("Generated Answer:", final_answer)
        
        return final_answer

# Initialize the model

