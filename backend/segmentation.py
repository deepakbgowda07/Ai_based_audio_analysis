"""
Topic segmentation module using semantic embeddings
Splits transcripts into semantic topics using sentence transformers
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationService:
    """Service for segmenting transcripts into topics"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", window_size=3):
        """
        Initialize the segmentation service
        
        Args:
            model_name: Sentence transformer model to use
            window_size: Number of sentences in context window
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.window_size = window_size
    
    @staticmethod
    def _moving_average(data, k=3):
        """Apply moving average smoothing"""
        return np.convolve(data, np.ones(k) / k, mode='same')
    
    @staticmethod
    def _time_to_seconds(time_str):
        """Convert HH:MM:SS to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
            else:
                return int(parts[0])
        except:
            return 0
    
    @staticmethod
    def _seconds_to_time(seconds):
        """Convert seconds to HH:MM:SS format"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    
    def segment_transcript(self, transcript_segments, min_topic_size=3, similarity_percentile=20):
        """
        Segment transcript into topics based on semantic similarity
        
        Args:
            transcript_segments: List of transcript segments with text and timestamps
            min_topic_size: Minimum number of segments per topic
            similarity_percentile: Percentile threshold for detecting boundaries
            
        Returns:
            List of topic segments with start/end times and combined text
        """
        if len(transcript_segments) < 5:
            logger.warning("Not enough segments for meaningful segmentation")
            # Return single topic if not enough segments
            return [{
                "start_time": transcript_segments[0]["timestamp"],
                "start_seconds": transcript_segments[0]["start_seconds"],
                "end_time": transcript_segments[-1]["timestamp"],
                "end_seconds": transcript_segments[-1]["end_seconds"],
                "content": " ".join(seg["text"] for seg in transcript_segments)
            }]
        
        logger.info(f"Segmenting {len(transcript_segments)} transcript segments")
        
        # Create context windows
        window_texts = [
            " ".join(
                transcript_segments[j]["text"]
                for j in range(i, min(i + self.window_size, len(transcript_segments)))
            )
            for i in range(len(transcript_segments))
        ]
        
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(window_texts, show_progress_bar=False)
        
        # Compute similarity between adjacent windows
        logger.info("Computing similarity scores...")
        similarities = [
            cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]
        similarities = np.array(similarities)
        
        # Smooth the similarity curve
        smoothed_similarities = self._moving_average(similarities, k=3)
        
        # Detect boundaries where similarity drops below threshold
        threshold = np.percentile(smoothed_similarities, similarity_percentile)
        logger.info(f"Similarity threshold: {threshold:.4f}")
        
        raw_boundaries = [
            i + 1 for i, sim in enumerate(smoothed_similarities)
            if sim < threshold
        ]
        
        # Filter boundaries to maintain minimum topic size
        boundaries = []
        last_boundary = 0
        for b in raw_boundaries:
            if b - last_boundary >= min_topic_size:
                boundaries.append(b)
                last_boundary = b
        
        logger.info(f"Detected {len(boundaries)} topic boundaries")
        
        # Group segments into topics
        topics = []
        current_topic = []
        boundary_set = set(boundaries)
        
        for i, seg in enumerate(transcript_segments):
            if i in boundary_set and current_topic:
                topics.append(current_topic)
                current_topic = []
            current_topic.append(seg)
        
        if current_topic:
            topics.append(current_topic)
        
        # Create structured output
        structured_topics = [
            {
                "start_time": topic[0]["timestamp"],
                "start_seconds": topic[0]["start_seconds"],
                "end_time": topic[-1]["timestamp"],
                "end_seconds": topic[-1]["end_seconds"],
                "content": " ".join(seg["text"] for seg in topic)
            }
            for topic in topics
        ]
        
        logger.info(f"Created {len(structured_topics)} topics")
        return structured_topics
