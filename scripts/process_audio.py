import os
import glob
import csv
import numpy as np
import librosa
import soundfile as sf

# ============ 配置 ============

# 原始 wav 目录：把那 50 首放这里
RAW_DIR = "data/raw"

# 输出目录
RAW_DIR = "../data/processed/raint/wav"  # 如果没跑 ffmpeg，就写 "../data/raw"
OUT_AUDIO_DIR = "../data/processed/audio_segments"
META_PATH = "../data/processed/metadata.csv"

# 采样率：改成 ACE-Step 基础模型的采样率
TARGET_SR = 24000

# 每段长度 / 重叠秒数
CHUNK_SEC = 10.0
OVERLAP_SEC = 2.0

# 静音裁剪阈值（不想裁剪可设 None）
SILENCE_TOP_DB = 30  # None 表示不做静音裁剪


# ============ 辅助函数 ============

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)


def unify_wavs():
    """
    把 data/raw/*.wav 统一到 TARGET_SR + mono，
    存到 data/processed/unified/ 下。
    """
    wav_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.wav")))
    if not wav_files:
        print(f"[ERROR] RAW_DIR={RAW_DIR} 下没有 wav 文件")
        return

    print(f"[INFO] 发现 {len(wav_files)} 个原始 wav 文件，开始统一采样率/声道")
    for i, wav_path in enumerate(wav_files):
        track_name = os.path.basename(wav_path)
        out_path = os.path.join(RAW_DIR, track_name)

        print(f"[INFO] ({i+1}/{len(wav_files)}) 统一处理: {wav_path} -> {out_path}")
        # librosa.load 会自动重采样 + 转 mono
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        sf.write(out_path, y, TARGET_SR, subtype="PCM_16")

    print("[INFO] 统一采样率/声道 完成")


def slice_wavs_and_make_metadata():
    """
    从 OUT_UNIFIED_DIR 读取 wav，切片，保存片段，并写 metadata.csv
    """
    wav_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.wav")))
    if not wav_files:
        print(f"[ERROR] OUT_UNIFIED_DIR={RAW_DIR} 下没有统一后的 wav 文件")
        return

    print(f"[INFO] 发现 {len(wav_files)} 个统一后的 wav 文件，开始切片")

    meta_rows = []
    segment_id = 0

    for idx, wav_path in enumerate(wav_files):
        track_name = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"[INFO] ({idx+1}/{len(wav_files)}) 处理曲目: {track_name}")

        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        if y.size == 0:
            print(f"[WARN] {wav_path} 为空，跳过")
            continue

        # 去前后静音
        if SILENCE_TOP_DB is not None:
            intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB)
            if len(intervals) > 0:
                parts = [y[start:end] for start, end in intervals]
                y = np.concatenate(parts)
            else:
                # 全静音，跳过
                print(f"[WARN] {wav_path} 静音过多，跳过")
                continue

        duration = len(y) / TARGET_SR
        if duration < 1.0:
            print(f"[WARN] {wav_path} 有效时长 < 1 秒，跳过")
            continue

        step = CHUNK_SEC - OVERLAP_SEC
        start_t = 0.0

        local_seg_count = 0

        while start_t + CHUNK_SEC <= duration:
            s = int(start_t * TARGET_SR)
            e = int((start_t + CHUNK_SEC) * TARGET_SR)
            chunk = y[s:e]

            seg_filename = f"{track_name}_seg{segment_id:05d}.wav"
            seg_path = os.path.join(OUT_AUDIO_DIR, seg_filename)

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
            local_seg_count += 1
            start_t += step

        print(f"[INFO] 曲目 {track_name} 生成片段数: {local_seg_count}")

    # 写 metadata.csv
    if meta_rows:
        fieldnames = [
            "segment_id",
            "audio_path",
            "track_name",
            "start_sec",
            "end_sec",
            "duration_sec",
            "instrument",
            "style",
            "prompt",
        ]
        with open(META_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in meta_rows:
                writer.writerow(row)

        print(f"[INFO] 写入 metadata: {META_PATH}, 共 {len(meta_rows)} 条片段")
    else:
        print("[WARN] 没有生成任何片段，metadata.csv 未写入")


def main():
    ensure_dirs()
    unify_wavs()
    slice_wavs_and_make_metadata()


if __name__ == "__main__":
    main()
