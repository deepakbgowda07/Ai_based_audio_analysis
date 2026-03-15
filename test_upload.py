import requests
import json

url = 'http://localhost:5000/api/upload'
audio_file = r'audio_dataset\Txt_Wav\clean_audio1.wav'

with open(audio_file, 'rb') as f:
    files = {'file': f}
    try:
        print('Uploading audio:', audio_file)
        r = requests.post(url, files=files, timeout=180)
        print(f'Status: {r.status_code}')
        
        if r.status_code == 200:
            print('✓ SUCCESS!')
            data = r.json()
            print(f'  Transcript segments: {len(data.get("transcript", []))}')
            print(f'  Analysis segments: {len(data.get("segments", []))}')
            perf = data.get('performance', {})
            print(f'  Preprocessing: {perf.get("preprocessing")}s')
            print(f'  Transcription: {perf.get("transcription")}s')
            print(f'  Embedding: {perf.get("embedding")}s')
            print(f'  Segmentation: {perf.get("segmentation")}s')
            print(f'  Sentiment: {perf.get("sentiment")}s')
        else:
            error_data = r.json()
            print(f'✗ ERROR: {error_data.get("error", "Unknown error")}')
            
    except Exception as e:
        print(f'Error: {e}')
