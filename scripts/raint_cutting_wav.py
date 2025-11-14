"""
Script to cut raint wav files into fixed-length segments with overlap,
remove silence, and save metadata.
分割处理 raint 数据集的 wav 文件，去除静音部分，保存为固定长度的音频片段，并生成元数据文件。
"""

import os
import glob
import soundfile as sf
import csv
import numpy as np
import librosa

RAW_DIR = "../data/processed/raint/wav"  # 如果没跑 ffmpeg，就写 "../data/raw"
OUT_AUDIO_DIR = "../data/processed/audio_segments"
META_PATH = "../data/processed/metadata.csv"

TARGET_SR = 24000
CHUNK_SEC = 10.0
OVERLAP_SEC = 2.0

os.makedirs(OUT_AUDIO_DIR, exist_ok=True)

meta_rows = []
segment_id = 0

wav_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.wav")))
print(f"Found {len(wav_files)} wav files in {RAW_DIR}")

for wav_path in wav_files:
    track_name = os.path.splitext(os.path.basename(wav_path))[0]
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    print(f"Loaded {wav_path} with sample rate {sr}")

    intervals = librosa.effects.split(y, top_db=30)
    if len(intervals) > 0:
        parts = [y[s:e] for s, e in intervals if e > s]
        if len(parts) == 1:
            y = parts[0]
        elif len(parts) > 1:
            y = np.concatenate(parts)
    duration = len(y) / TARGET_SR
    if duration < 1.0:
        continue

    step = CHUNK_SEC - OVERLAP_SEC
    start_t = 0.0
    while start_t + CHUNK_SEC <= duration:
        s = int(start_t * TARGET_SR)
        e = int((start_t + CHUNK_SEC) * TARGET_SR)
        chunk = y[s:e]

        seg_filename = f"{track_name}_seg{segment_id:05d}.wav"
        seg_path = os.path.join(OUT_AUDIO_DIR, seg_filename)
        print(f"Writing segment: {seg_path}")

        sf.write(seg_path, chunk, TARGET_SR, subtype="PCM_16")

        meta_rows.append({
            "segment_id": segment_id,
            "audio_path": seg_path,
            "track_name": track_name,
            "start_sec": round(start_t, 3),
            "end_sec": round(start_t + CHUNK_SEC, 3),
            "duration_sec": round(CHUNK_SEC, 3),
            "instrument": "马头琴",
            "style": "蒙古族 传统 器乐 独奏",
            "prompt": "马头琴独奏，蒙古族传统器乐风格，干净清晰的录音"
        })

        segment_id += 1
        start_t += step

with open(META_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "segment_id",
            "audio_path",
            "track_name",
            "start_sec",
            "end_sec",
            "duration_sec",
            "instrument",
            "style",
            "prompt",
        ],
    )
    writer.writeheader()
    writer.writerows(meta_rows)

print(f"Saved {len(meta_rows)} segments metadata to {META_PATH}")