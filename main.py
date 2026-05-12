from pathlib import Path
import time

import numpy as np
import pandas as pd


INPUT_CLEAN = Path("clean_tracks.csv")
OUTPUT_SUMMARY_CSV = Path("interp_k_summary.csv")
K_VALUES = list(range(0, 300, 10))
METHODS = ["linear", "slerp"]


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return radius * 2.0 * np.arcsin(np.sqrt(a))


def unit_vec(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z])


def slerp_between(lat_left, lon_left, lat_right, lon_right, alpha):
    alpha = np.asarray(alpha, dtype=float)
    left_vec = unit_vec(lat_left, lon_left)
    right_vec = unit_vec(lat_right, lon_right)

    dot = np.clip(np.sum(left_vec * right_vec, axis=1), -1.0, 1.0)
    omega = np.arccos(dot)
    sin_omega = np.sin(omega)

    result = np.empty_like(left_vec)
    near_linear = np.isclose(sin_omega, 0.0)
    result[near_linear] = left_vec[near_linear]

    valid = ~near_linear
    if np.any(valid):
        w_left = np.sin((1.0 - alpha[valid]) * omega[valid]) / sin_omega[valid]
        w_right = np.sin(alpha[valid] * omega[valid]) / sin_omega[valid]
        result[valid] = w_left[:, None] * left_vec[valid] + w_right[:, None] * right_vec[valid]

    norm = np.linalg.norm(result, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    result = result / norm

    lat = np.degrees(np.arcsin(np.clip(result[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(result[:, 1], result[:, 0]))
    return lat, lon


def linear_between(lat_left, lon_left, lat_right, lon_right, alpha):
    pred_lat = lat_left + alpha * (lat_right - lat_left)
    pred_lon = lon_left + alpha * (lon_right - lon_left)
    return pred_lat, pred_lon


def evaluate_track_for_k(t, lat, lon, k):
    n_points = len(t)
    if n_points < 2 * k + 1:
        return None

    center_idx = np.arange(k, n_points - k)
    left_idx = center_idx - k
    right_idx = center_idx + k

    t_left = t[left_idx]
    t_center = t[center_idx]
    t_right = t[right_idx]
    span = t_right - t_left

    valid_mask = span > 0
    if not np.any(valid_mask):
        return {
            "linear": np.array([], dtype=float),
            "slerp": np.array([], dtype=float),
            "invalid_samples": int(center_idx.size),
            "valid_samples": 0,
        }

    left_idx = left_idx[valid_mask]
    center_idx = center_idx[valid_mask]
    right_idx = right_idx[valid_mask]
    t_left = t_left[valid_mask]
    t_center = t_center[valid_mask]
    t_right = t_right[valid_mask]
    alpha = (t_center - t_left) / (t_right - t_left)

    lat_left = lat[left_idx]
    lon_left = lon[left_idx]
    lat_right = lat[right_idx]
    lon_right = lon[right_idx]
    lat_true = lat[center_idx]
    lon_true = lon[center_idx]

    start_time = time.perf_counter()
    lin_lat, lin_lon = linear_between(lat_left, lon_left, lat_right, lon_right, alpha)
    linear_elapsed = time.perf_counter() - start_time

    start_time = time.perf_counter()
    slerp_lat, slerp_lon = slerp_between(lat_left, lon_left, lat_right, lon_right, alpha)
    slerp_elapsed = time.perf_counter() - start_time

    return {
        "linear": haversine(lat_true, lon_true, lin_lat, lin_lon),
        "slerp": haversine(lat_true, lon_true, slerp_lat, slerp_lon),
        "invalid_samples": int(np.count_nonzero(~valid_mask)),
        "valid_samples": int(np.count_nonzero(valid_mask)),
        "timings": {
            "linear": linear_elapsed,
            "slerp": slerp_elapsed,
        },
    }


def build_summary_rows(error_store, timing_store):
    rows = []
    for k in K_VALUES:
        for method in METHODS:
            error_list = error_store[(k, method)]
            if not error_list:
                continue

            all_errors = np.concatenate(error_list)
            total_seconds = timing_store[(k, method)]["seconds"]
            total_points = timing_store[(k, method)]["points"]
            time_per_1000_ms = 1000.0 * total_seconds / total_points * 1000.0 if total_points else np.nan
            rows.append(
                {
                    "k": k,
                    "Method": method,
                    "SampleCount": int(all_errors.size),
                    "MAE_m": float(np.mean(all_errors)),
                    "RMSE_m": float(np.sqrt(np.mean(all_errors ** 2))),
                    "Median_m": float(np.median(all_errors)),
                    "P95_m": float(np.percentile(all_errors, 95)),
                    "TimePer1000Pts_ms": float(time_per_1000_ms),
                }
            )
    return pd.DataFrame(rows)


def build_overall_timing_rows(total_timing_store):
    rows = []
    for method in METHODS:
        total_seconds = total_timing_store[method]["seconds"]
        total_points = total_timing_store[method]["points"]
        time_per_1000_ms = 1000.0 * total_seconds / total_points * 1000.0 if total_points else np.nan
        rows.append(
            {
                "Method": method,
                "TotalPoints": int(total_points),
                "TotalTime_s": float(total_seconds),
                "TimePer1000Pts_ms": float(time_per_1000_ms),
            }
        )
    return pd.DataFrame(rows)


def load_tracks(input_path):
    df = pd.read_csv(input_path, parse_dates=["Timestamp"])
    df["t"] = df["Timestamp"].astype("int64") // 10**9
    return df.groupby(["MMSI", "SegmentId"], sort=False)


def main():
    max_k = max(K_VALUES)
    min_track_points = 2 * max_k + 1

    error_store = {(k, method): [] for k in K_VALUES for method in METHODS}
    timing_store = {(k, method): {"seconds": 0.0, "points": 0} for k in K_VALUES for method in METHODS}
    total_timing_store = {method: {"seconds": 0.0, "points": 0} for method in METHODS}
    stats = {
        "total_tracks": 0,
        "tracks_too_short_for_all_k": 0,
        "invalid_samples": 0,
        "valid_samples": 0,
    }

    for (_, _), group in load_tracks(INPUT_CLEAN):
        stats["total_tracks"] += 1

        track = group.sort_values("t")
        t = track["t"].to_numpy(dtype=np.int64)
        lat = track["Latitude"].to_numpy(dtype=float)
        lon = track["Longitude"].to_numpy(dtype=float)

        if len(t) < min_track_points:
            stats["tracks_too_short_for_all_k"] += 1

        for k in K_VALUES:
            result = evaluate_track_for_k(t, lat, lon, k)
            if result is None:
                continue

            stats["invalid_samples"] += result["invalid_samples"]
            stats["valid_samples"] += result["valid_samples"]

            for method in METHODS:
                if result[method].size > 0:
                    error_store[(k, method)].append(result[method])
                    timing_store[(k, method)]["seconds"] += result["timings"][method]
                    timing_store[(k, method)]["points"] += result[method].size
                    total_timing_store[method]["seconds"] += result["timings"][method]
                    total_timing_store[method]["points"] += result[method].size

    summary_df = build_summary_rows(error_store, timing_store)
    if summary_df.empty:
        raise RuntimeError("没有生成任何插值误差结果，请检查 clean_tracks.csv 是否包含足够长的轨迹段。")

    overall_timing_df = build_overall_timing_rows(total_timing_store)
    summary_df = summary_df.sort_values(["Method", "k"]).reset_index(drop=True)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    with pd.option_context("display.float_format", "{:.12f}".format):
        print(summary_df.to_string(index=False))
        print("\n整体方法耗时统计:")
        print(overall_timing_df.to_string(index=False))
    print(f"\n总轨迹段数: {stats['total_tracks']}")
    print(f"不足以覆盖全部 k 的轨迹段数: {stats['tracks_too_short_for_all_k']}")
    print(f"有效插值样本数: {stats['valid_samples']}")
    print(f"跳过的无效样本数: {stats['invalid_samples']}")
    print(f"\n汇总结果已导出至: {OUTPUT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
