from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


RAW_INPUT = Path("aisdk-2025-02-27.csv")
CLEAN_OUTPUT = Path("clean_tracks.csv")
CHUNK_SIZE = 200_000
MAX_GAP_SECONDS = 30
MIN_POINTS = 1024
MAX_SPEED_MPS = 100.0
RAW_COLUMNS = ["Timestamp", "MMSI", "Latitude", "Longitude"]


def normalize_col(name: str) -> str:
    return name.lstrip("# ").strip()


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(a))


def clean_raw_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns=normalize_col)
    chunk = chunk[RAW_COLUMNS].copy()
    chunk["Timestamp"] = pd.to_datetime(chunk["Timestamp"], dayfirst=True, errors="coerce")
    chunk = chunk.dropna()
    chunk = chunk[chunk["Latitude"].between(-90, 90) & chunk["Longitude"].between(-180, 180)]
    chunk["MMSI"] = chunk["MMSI"].astype(np.int64)
    return chunk


def load_raw_tracks():
    buffers = defaultdict(list)
    reader = pd.read_csv(
        RAW_INPUT,
        usecols=lambda col: normalize_col(col) in RAW_COLUMNS,
        chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        chunk = clean_raw_chunk(chunk)
        if chunk.empty:
            continue

        for mmsi, group in chunk.groupby("MMSI", sort=False):
            t = (group["Timestamp"].astype("int64") // 10**9).to_numpy()
            lat = group["Latitude"].to_numpy()
            lon = group["Longitude"].to_numpy()
            buffers[int(mmsi)].append(np.column_stack([t, lat, lon]))
    return buffers


def dedup(seg):
    t = seg[:, 0].astype(np.int64)
    unique_t, inverse_idx = np.unique(t, return_inverse=True)
    lat = np.zeros_like(unique_t, dtype=float)
    lon = np.zeros_like(unique_t, dtype=float)
    np.add.at(lat, inverse_idx, seg[:, 1])
    np.add.at(lon, inverse_idx, seg[:, 2])
    counts = np.bincount(inverse_idx)
    return unique_t, lat / counts, lon / counts


def greedy_speed_filter(t, lat, lon):
    keep_indices = [0]
    last_kept = 0

    for idx in range(1, len(t)):
        dt = float(t[idx] - t[last_kept])
        if dt <= 0:
            continue

        dist = float(haversine(lat[last_kept], lon[last_kept], lat[idx], lon[idx]))
        if dist <= dt * MAX_SPEED_MPS:
            keep_indices.append(idx)
            last_kept = idx

    return np.array(keep_indices, dtype=np.int64)


def filter_segment_outliers(t, lat, lon):
    if len(t) <= 2:
        return t, lat, lon, 0

    forward_keep = greedy_speed_filter(t, lat, lon)
    reversed_keep = greedy_speed_filter(t[::-1], lat[::-1], lon[::-1])
    backward_keep = (len(t) - 1 - reversed_keep)[::-1]

    if len(backward_keep) > len(forward_keep):
        keep = backward_keep
    else:
        keep = forward_keep

    removed_points = int(len(t) - len(keep))
    return t[keep], lat[keep], lon[keep], removed_points


def generate_clean_tracks(buffers):
    rows = []
    seg_id = 1
    stats = {
        "raw_segments": 0,
        "segments_too_short_before_filter": 0,
        "segments_dropped_after_filter": 0,
        "removed_points": 0,
        "kept_points": 0,
    }

    for mmsi, pieces in buffers.items():
        track = np.concatenate(pieces)
        track = track[np.argsort(track[:, 0])]
        splits = np.where(np.diff(track[:, 0]) > MAX_GAP_SECONDS)[0] + 1

        for seg in np.split(track, splits):
            stats["raw_segments"] += 1
            t, lat, lon = dedup(seg)
            if len(t) < MIN_POINTS:
                stats["segments_too_short_before_filter"] += 1
                continue

            t, lat, lon, removed_points = filter_segment_outliers(t, lat, lon)
            stats["removed_points"] += removed_points

            if len(t) < 2:
                stats["segments_dropped_after_filter"] += 1
                continue

            stats["kept_points"] += int(len(t))
            rows.append(
                pd.DataFrame(
                    {
                        "MMSI": mmsi,
                        "SegmentId": seg_id,
                        "Timestamp": pd.to_datetime(t, unit="s"),
                        "Latitude": lat,
                        "Longitude": lon,
                    }
                )
            )
            seg_id += 1

    if rows:
        final = pd.concat(rows, ignore_index=True)
    else:
        final = pd.DataFrame(columns=["MMSI", "SegmentId", "Timestamp", "Latitude", "Longitude"])

    final.to_csv(CLEAN_OUTPUT, index=False)
    print(f"清洗完成！有效轨迹段：{seg_id - 1}")
    print(f"原始切分段数：{stats['raw_segments']}")
    print(f"删点前长度不足的段数：{stats['segments_too_short_before_filter']}")
    print(f"删点后被丢弃的段数：{stats['segments_dropped_after_filter']}")
    print(f"删除的异常点数：{stats['removed_points']}")
    print(f"保留的点数：{stats['kept_points']}")


if __name__ == "__main__":
    tracks = load_raw_tracks()
    generate_clean_tracks(tracks)