"""
Analysis module for segment enrichment
Extracts keywords, summaries, and sentiment from transcript segments
"""

import logging
from keybert import KeyBERT
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.pipelines.text2text_generation import SummarizationPipeline
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisService:
    """Service for analyzing and enriching transcript segments"""
    
    def __init__(self):
        """Initialize the analysis service with required models"""
        logger.info("Initializing analysis models...")
        self.kw_model = KeyBERT()
        logger.info("Loading summarization model...")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = SummarizationPipeline(model=model, tokenizer=tokenizer)
        logger.info("Analysis service initialized")
    
    @staticmethod
    def extract_keywords(text, top_n=5):
        """
        Extract keywords from text using KeyBERT
        
        Args:
            text: Input text to extract keywords from
            top_n: Number of keywords to extract
            
        Returns:
            List of keyword strings
        """
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n
        )
        return [k[0] for k in keywords]
    
    def generate_summary(self, text, max_length=40, min_length=15):
        """
        Generate a summary of the text using BART
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            Summary text
        """
        # Truncate if too long (BART has token limits)
        text = text[:1024]
        
        if len(text.split()) < min_length:
            return text
        
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return result[0]["summary_text"]
    
    @staticmethod
    def get_sentiment(text):
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        blob = TextBlob(text)
        return round(blob.sentiment.polarity, 3)
    
    def analyze_segments(self, segments, episode_id="episode_1"):
        """
        Analyze and enrich transcript segments
        
        Args:
            segments: List of segmented topics with content
            episode_id: ID for the episode
            
        Returns:
            Structured JSON output with all analyses
        """
        logger.info(f"Analyzing {len(segments)} segments...")
        
        processed_segments = []
        
        for idx, segment in enumerate(segments):
            logger.info(f"Processing segment {idx + 1}/{len(segments)}")
            
            text = segment["content"]
            
            # Extract analysis
            keywords = self.extract_keywords(text, top_n=5)
            summary = self.generate_summary(text)
            sentiment = self.get_sentiment(text)
            
            processed_segments.append({
                "segment_id": f"seg_{idx + 1}",
                "start_time": segment["start_seconds"],
                "end_time": segment["end_seconds"],
                "start_time_formatted": segment["start_time"],
                "end_time_formatted": segment["end_time"],
                "text": text,
                "keywords": keywords,
                "summary": summary,
                "sentiment_score": sentiment
            })
        
        # Calculate total duration
        total_duration = max(
            seg["end_seconds"] for seg in segments
        ) if segments else 0
        
        # Create final output
        final_output = {
            "episode_id": episode_id,
            "duration": total_duration,
            "num_segments": len(processed_segments),
            "segments": processed_segments
        }
        
        logger.info("Analysis complete")
        return final_output
