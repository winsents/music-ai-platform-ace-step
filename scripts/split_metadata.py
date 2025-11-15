import os
import csv
import random
from collections import defaultdict

META_PATH = "../data/processed/metadata.csv"
TRAIN_META_PATH = "../data/processed/metadata_train.csv"
VAL_META_PATH = "../data/processed/metadata_val.csv"
TEST_META_PATH = "../data/processed/metadata_test.csv"

# 按曲目数量划分比例（train : val : test）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)


def read_metadata():
    if not os.path.isfile(META_PATH):
        print(f"[ERROR] 找不到 {META_PATH}")
        return None, None

    with open(META_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"[INFO] 读取 metadata: {META_PATH}, 共 {len(rows)} 条片段")
    return rows, fieldnames


def group_by_track(rows):
    groups = defaultdict(list)
    for row in rows:
        track_name = row.get("track_name", "")
        groups[track_name].append(row)

    print(f"[INFO] 共有 {len(groups)} 首曲目 (track_name)")
    return groups


def split_tracks(track_names):
    track_names = list(track_names)
    random.shuffle(track_names)

    n = len(track_names)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # 剩余给 test
    n_test = n - n_train - n_val

    train_tracks = track_names[:n_train]
    val_tracks = track_names[n_train:n_train + n_val]
    test_tracks = track_names[n_train + n_val:]

    print(f"[INFO] 曲目数量划分: train={len(train_tracks)}, val={len(val_tracks)}, test={len(test_tracks)}")
    return set(train_tracks), set(val_tracks), set(test_tracks)


def write_split_csv(groups, fieldnames, train_set, val_set, test_set):
    train_rows, val_rows, test_rows = [], [], []

    for track_name, rows in groups.items():
        if track_name in train_set:
            train_rows.extend(rows)
        elif track_name in val_set:
            val_rows.extend(rows)
        elif track_name in test_set:
            test_rows.extend(rows)
        else:
            # 理论上不会出现
            print(f"[WARN] track_name {track_name} 未被分配到任何集合")

    def write_csv(path, rows):
        if not rows:
            print(f"[WARN] {path} 没有任何行，将不写入")
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[INFO] 写入 {path}, 条数={len(rows)}")

    write_csv(TRAIN_META_PATH, train_rows)
    write_csv(VAL_META_PATH, val_rows)
    write_csv(TEST_META_PATH, test_rows)


def main():
    rows, fieldnames = read_metadata()
    if rows is None:
        return

    groups = group_by_track(rows)
    track_names = list(groups.keys())

    if len(track_names) < 2:
        print("[WARN] 曲目数量太少，没必要拆 train/val/test")
        return

    train_set, val_set, test_set = split_tracks(track_names)
    write_split_csv(groups, fieldnames, train_set, val_set, test_set)


if __name__ == "__main__":
    main()
