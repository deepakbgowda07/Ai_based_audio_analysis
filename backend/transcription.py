"""
Transcription module using faster-whisper
Converts audio files to timestamped transcripts
"""

import os
import re
import logging
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_SIZE = "medium.en"
FILLER_WORDS = [
    "uh", "um", "you know", "like", "i mean",
    "sort of", "kind of", "basically", "actually"
]

class TranscriptionService:
    """Service for transcribing audio files using Whisper"""
    
    def __init__(self, model_size=MODEL_SIZE):
        """
        Initialize the transcription service
        
        Args:
            model_size: The Whisper model size to use
        """
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16"
        )
        self.model_size = model_size
    
    @staticmethod
    def format_timestamp(seconds):
        """Convert seconds to HH:MM:SS format"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs:02}:{mins:02}:{secs:02}"
    
    @staticmethod
    def remove_fillers(text):
        """Remove common filler words from text"""
        pattern = r"\b(" + "|".join(FILLER_WORDS) + r")\b"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    @staticmethod
    def clean_text(text):
        """Clean text by removing fillers and extra whitespace"""
        text = TranscriptionService.remove_fillers(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    @staticmethod
    def split_into_sentences(text):
        """Split text into sentences"""
        return re.split(r'(?<=[.!?]) +', text)
    
    def transcribe(self, audio_file_path):
        """
        Transcribe an audio file
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            List of transcript segments with timestamps
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        logger.info(f"Transcribing: {audio_file_path}")
        
        segments, info = self.model.transcribe(
            audio_file_path,
            beam_size=5,
            language="en",
            vad_filter=True
        )
        
        # Process segments
        transcript_segments = []
        for segment in segments:
            cleaned = self.clean_text(segment.text)
            sentences = self.split_into_sentences(cleaned)
            
            for sentence in sentences:
                if sentence.strip():
                    timestamp = self.format_timestamp(segment.start)
                    transcript_segments.append({
                        "timestamp": timestamp,
                        "start_seconds": segment.start,
                        "end_seconds": segment.end,
                        "text": sentence.strip()
                    })
        
        logger.info(f"Transcription complete. Generated {len(transcript_segments)} segments")
        return transcript_segments
