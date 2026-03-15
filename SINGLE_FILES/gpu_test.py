from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda")
print("Model loaded on GPU successfully!")
