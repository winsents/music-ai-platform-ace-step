import os
import csv
from pathlib import Path

from pydub import AudioSegment  # 需要 pip install pydub

META_PATH = "../data/processed/metadata_train.csv"
OUT_DIR = "../data/ace_data"

os.makedirs(OUT_DIR, exist_ok=True)

def wav_to_mp3(src_wav, dst_mp3):
    audio = AudioSegment.from_file(src_wav, format="wav")
    audio.export(dst_mp3, format="mp3", bitrate="192k")

def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] rows in metadata_train: {len(rows)}")

    for i, row in enumerate(rows):
        seg_id = int(row["segment_id"])
        wav_path = row["audio_path"]
        wav_path = wav_path.strip()

        if not os.path.isfile(wav_path):
            print(f"[WARN] not found: {wav_path}, skip")
            continue

        key = f"morin_train_{seg_id:06d}"

        mp3_path = os.path.join(OUT_DIR, f"{key}.mp3")
        prompt_path = os.path.join(OUT_DIR, f"{key}_prompt.txt")
        lyrics_path = os.path.join(OUT_DIR, f"{key}_lyrics.txt")

        print(f"[INFO] ({i+1}/{len(rows)}) {wav_path} -> {mp3_path}")

        # 1) wav -> mp3
        wav_to_mp3(wav_path, mp3_path)

        # 2) prompt 标签：用逗号分隔的英文/中文标签，适配 ACE-Step 的期望格式
        tags = [
            "morin khuur",          # 马头琴
            "solo",                 # 独奏
            "mongolian traditional",
            "instrumental",
            "clean recording",
            "slow to mid tempo",
        ]
        with open(prompt_path, "w", encoding="utf-8") as pf:
            pf.write(", ".join(tags))

        # 3) 器乐，没有歌词，写个占位或留空都行
        with open(lyrics_path, "w", encoding="utf-8") as lf:
            lf.write("[Instrumental]\n")

if __name__ == "__main__":
    main()
