import os
import subprocess

# -------------------------
# CONFIG
# -------------------------

INPUT_FILE = "audio_dataset/podcasts/clean_audio2.mp3"
OUTPUT_FOLDER = "audio_dataset/Txt_Wav"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# Conversion
# -------------------------

def convert_to_wav(input_file):
    filename = os.path.basename(input_file)
    output_name = os.path.splitext(filename)[0] + ".wav"
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    cmd = f'ffmpeg -i "{input_file}" -ac 1 -ar 16000 "{output_path}" -y'
    subprocess.run(cmd, shell=True)

    print(f"Converted: {filename} → {output_name}")

if __name__ == "__main__":
    convert_to_wav(INPUT_FILE)

print("All files converted.")