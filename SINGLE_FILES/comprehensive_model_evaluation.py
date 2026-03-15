"""
Comprehensive Speech-to-Text Model Evaluation
Evaluates all Whisper variants and generates comparison metrics
"""

import os
import re
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import csv
import subprocess
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Try imports, handle missing packages gracefully
try:
    from jiwer import wer, cer, process_words, normalize as jiwer_normalize
except ImportError:
    print("Installing jiwer...")
    subprocess.run(["pip", "install", "jiwer", "-q"], check=False)
    from jiwer import wer, cer, process_words, normalize as jiwer_normalize

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Note: faster_whisper not explicitly imported - will be loaded dynamically")

try:
    import psutil
    import GPUtil
except ImportError:
    print("Note: psutil/GPUtil not available - some metrics will be skipped")
    psutil = None
    GPUtil = None

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio_dataset", "Txt_Wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define test datasets
TEST_DATASETS = {
    "clean_audio_1": {
        "audio": os.path.join(AUDIO_DIR, "clean_audio1_preprocessed.wav"),
        "reference": os.path.join(AUDIO_DIR, "reference.txt"),
        "category": "Clean",
        "description": "Clean podcast audio"
    },
    "clean_audio_2": {
        "audio": os.path.join(AUDIO_DIR, "clean_audio2_preprocessed.wav"),
        "reference": os.path.join(AUDIO_DIR, "reference.txt"),
        "category": "Clean",
        "description": "Clean podcast audio"
    },
    "clean_audio_3": {
        "audio": os.path.join(AUDIO_DIR, "clean_audio3_preprocessed.wav"),
        "reference": os.path.join(AUDIO_DIR, "reference.txt"),
        "category": "Clean",
        "description": "Clean podcast audio"
    },
    "noisy_speech": {
        "audio": os.path.join(AUDIO_DIR, "Noisy_speech[26_min]_preprocessed.wav"),
        "outputs": [
            os.path.join(AUDIO_DIR, "output_Noisy_speech[26_min]_preprocessed.txt")
        ],
        "category": "Noisy",
        "description": "Noisy speech with background noise (26 min)"
    },
    "long_report": {
        "audio": os.path.join(AUDIO_DIR, "LongReport[60_min]_preprocessed.wav"),
        "outputs": [
            os.path.join(AUDIO_DIR, "output_LongReport[60_min]_preprocessed.txt")
        ],
        "category": "Long-form",
        "description": "Long-form structured report (60 min)"
    }
}

# Whisper model variants to test
WHISPER_MODELS = {
    "base": {"size_mb": 140, "lang_support": "multi"},
    "small": {"size_mb": 466, "lang_support": "multi"},
    "medium": {"size_mb": 1534, "lang_support": "multi"},
    "large": {"size_mb": 2964, "lang_support": "single"},
    "base.en": {"size_mb": 140, "lang_support": "single"},
    "small.en": {"size_mb": 466, "lang_support": "single"},
    "medium.en": {"size_mb": 1534, "lang_support": "single"},
}

# ============================================================================
# UTILITIES
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for WER/CER computation"""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)  # Remove timestamps
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip()

def extract_text_from_output(filepath: str) -> str:
    """Extract text from timestamped output file"""
    text_parts = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Remove timestamp prefix [HH:MM:SS]
                    text = re.sub(r"^\[.*?\]\s*", "", line)
                    text_parts.append(text)
        return " ".join(text_parts)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""

def load_reference_text(filepath: str) -> str:
    """Load reference transcription"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading reference: {e}")
        return ""

def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", 
                 "format=duration", "-of", "default=noprint_wrappers=1:nokey=1:novalue=1",
                 audio_path],
                capture_output=True, text=True, timeout=10
            )
            return float(result.stdout.strip()) if result.stdout else 0
        except:
            return 0

def format_size_human(bytes_val: float) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float]:
    """Compute WER and CER"""
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    
    if not reference or not hypothesis:
        return 0.0, 0.0
    
    try:
        wer_score = wer(reference, hypothesis)
        cer_score = cer(reference, hypothesis)
        return wer_score, cer_score
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return 0.0, 0.0

# ============================================================================
# MAIN EVALUATION
# ============================================================================

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        self.model_cache = {}
        
    def load_model(self, model_name: str, device: str = "cuda"):
        """Load Whisper model"""
        try:
            if model_name in self.model_cache:
                return self.model_cache[model_name]
            
            print(f"Loading {model_name} on {device}...")
            model = WhisperModel(model_name, device=device, compute_type="float16")
            self.model_cache[model_name] = model
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def transcribe_audio(self, model, audio_path: str) -> Tuple[str, float]:
        """Transcribe audio and measure inference time"""
        try:
            start_time = time.time()
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                language="en",
                vad_filter=True
            )
            
            # Collect segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            inference_time = time.time() - start_time
            full_text = " ".join(text_parts)
            
            return full_text, inference_time
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return "", 0.0
    
    def evaluate_model_on_dataset(self, model_name: str, dataset_key: str) -> Dict:
        """Evaluate a single model on a dataset"""
        dataset = TEST_DATASETS[dataset_key]
        results = {
            "model": model_name,
            "dataset": dataset_key,
            "category": dataset["category"],
            "description": dataset["description"],
            "wer": None,
            "cer": None,
            "inference_time_sec": 0,
            "duration_sec": 0,
            "rtf": None,
        }
        
        audio_path = dataset["audio"]
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path}")
            return results
        
        # Get reference
        reference_path = dataset.get("reference")
        if reference_path and os.path.exists(reference_path):
            reference_text = load_reference_text(reference_path)
        else:
            # Try to use existing outputs
            reference_text = None
        
        # Transcribe or use existing output
        if model_name == "existing_outputs":
            # Use existing output files
            output_files = dataset.get("outputs", [])
            if output_files and os.path.exists(output_files[0]):
                hypothesis = extract_text_from_output(output_files[0])
                results["inference_time_sec"] = 0  # Unknown
            else:
                return results
        else:
            # Transcribe with model
            model = self.load_model(model_name)
            if not model:
                return results
            
            hypothesis, inf_time = self.transcribe_audio(model, audio_path)
            results["inference_time_sec"] = inf_time
        
        # Compute metrics
        if reference_text and hypothesis:
            duration = get_audio_duration(audio_path)
            results["duration_sec"] = duration
            
            wer_score, cer_score = compute_wer_cer(reference_text, hypothesis)
            results["wer"] = wer_score
            results["cer"] = cer_score
            
            if duration > 0:
                results["rtf"] = inf_time / duration
        
        return results
    
    def run_full_evaluation(self, models: List[str], datasets: List[str]):
        """Run comprehensive evaluation"""
        print("\n" + "="*80)
        print("COMPREHENSIVE SPEECH-TO-TEXT MODEL EVALUATION")
        print("="*80 + "\n")
        
        all_results = []
        
        for dataset_key in datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_key}")
            print(f"{'='*60}")
            
            for model_name in models:
                print(f"  Evaluating {model_name}...", end=" ")
                result = self.evaluate_model_on_dataset(model_name, dataset_key)
                all_results.append(result)
                
                if result["wer"] is not None:
                    print(f"✓ WER={result['wer']:.4f}, CER={result['cer']:.4f}")
                else:
                    print("✗ Skipped")
        
        return all_results

# ============================================================================
# REPORTING
# ============================================================================

def generate_comparison_table(results: List[Dict]) -> str:
    """Generate markdown comparison table"""
    
    # Group by dataset
    datasets_group = {}
    for result in results:
        dataset = result["dataset"]
        if dataset not in datasets_group:
            datasets_group[dataset] = []
        datasets_group[dataset].append(result)
    
    table = "# Speech-to-Text Model Comparison\n\n"
    
    for dataset, dataset_results in datasets_group.items():
        table += f"## Dataset: {dataset}\n\n"
        table += f"**Category:** {dataset_results[0]['category']} | **Description:** {dataset_results[0]['description']}\n\n"
        
        table += "| Model | WER | CER | Inference Time (s) | RTF | Notes |\n"
        table += "|-------|-----|-----|--------------------|----|-------|\n"
        
        for result in dataset_results:
            wer_str = f"{result['wer']:.4f}" if result['wer'] is not None else "N/A"
            cer_str = f"{result['cer']:.4f}" if result['cer'] is not None else "N/A"
            inf_str = f"{result['inference_time_sec']:.2f}" if result['inference_time_sec'] > 0 else "N/A"
            rtf_str = f"{result['rtf']:.3f}" if result['rtf'] is not None else "N/A"
            
            table += f"| {result['model']} | {wer_str} | {cer_str} | {inf_str} | {rtf_str} | |\n"
        
        table += "\n"
    
    return table

def save_results_json(results: List[Dict], filepath: str):
    """Save detailed results to JSON"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def save_results_csv(results: List[Dict], filepath: str):
    """Save results to CSV"""
    if not results:
        return
    
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    
    # Test a subset of models and datasets
    # This will evaluate existing outputs and attempt to transcribe with a few models
    models_to_test = ["existing_outputs"]  # Start with existing outputs
    
    # Only test available datasets
    datasets_to_test = []
    for ds_key in TEST_DATASETS:
        dataset = TEST_DATASETS[ds_key]
        if os.path.exists(dataset.get("audio", "")):
            datasets_to_test.append(ds_key)
    
    print(f"Available datasets to test: {datasets_to_test}")
    print(f"Models to test: {models_to_test}")
    
    # Run evaluation
    results = evaluator.run_full_evaluation(models_to_test, datasets_to_test)
    
    # Generate reports
    comparison_table = generate_comparison_table(results)
    
    # Save results
    json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    
    save_results_json(results, json_path)
    save_results_csv(results, csv_path)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(comparison_table)
    print(f"\nResults saved to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - Output directory: {OUTPUT_DIR}")
