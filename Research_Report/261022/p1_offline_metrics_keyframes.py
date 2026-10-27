#!/usr/bin/env python3
"""Offline IPN-Hand/CTPGesture metrics + misclassification keyframe contact sheets.

Inputs are the trusted CSV and pickle raw caches produced by the user's notebooks.
NEVER unpickle files from an untrusted source.
No VideoMamba checkpoint, GPU, or network connection is needed.

Example:
  python p1_offline_metrics_keyframes.py --dataset ctp \
    --isolated /kaggle/working/ctpgesture_videomamba_evaluation/isolated_predictions.csv \
    --cache /kaggle/working/ctpgesture_videomamba_evaluation/continuous_raw_cache.pkl \
    --out /kaggle/working/p1_ctp --video-root /kaggle/input

  python p1_offline_metrics_keyframes.py --dataset ipn \
    --isolated /kaggle/working/videomamba_ipn_isolated_test.csv \
    --cache /kaggle/working/videomamba_postprocess_raw_cache.pkl \
    --out /kaggle/working/p1_ipn --video-root /kaggle/input
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)


def metric_pack(y, yp, labels=None):
    y = np.asarray(y, dtype=np.int64)
    yp = np.asarray(yp, dtype=np.int64)
    if len(y) == 0:
        return {}
    args = dict(zero_division=0)
    if labels is not None:
        args["labels"] = labels
    return {
        "n": len(y),
        "accuracy": float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "macro_precision": float(precision_score(y, yp, average="macro", **args)),
        "macro_recall": float(recall_score(y, yp, average="macro", **args)),
        "macro_f1": float(f1_score(y, yp, average="macro", **args)),
        "weighted_f1": float(f1_score(y, yp, average="weighted", **args)),
    }


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def segments_from_labels(labels, ignore_zero=True):
    a = np.asarray(labels, dtype=np.int64)
    if a.size == 0:
        return []
    changes = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1, len(a)]
    out = []
    for start, end in zip(changes[:-1], changes[1:]):
        v = int(a[start])
        if not (ignore_zero and v == 0):
            out.append(dict(start=int(start), end=int(end - 1), label=v))
    return out


def edit_distance(a, b):
    # Standard Levenshtein with unit substitution, insertion and deletion.
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1,
                         prev[j - 1] + (x != y))
        prev = cur
    return int(prev[-1])


def frame_probs_from_cache(entry):
    p = np.asarray(entry["P_raw"], dtype=np.float32)
    starts = np.asarray(entry["window_starts"], dtype=np.int64)
    ends = np.asarray(entry["window_ends"], dtype=np.int64)
    gt = np.asarray(entry["gt_labels"], dtype=np.int64)
    if p.ndim != 2 or len(p) != len(starts) or len(starts) != len(ends):
        raise ValueError("Unexpected probability/window dimensions")
    sums = np.zeros((len(gt), p.shape[1]), dtype=np.float64)
    counts = np.zeros(len(gt), dtype=np.int32)
    for prob, start, end in zip(p, starts, ends):
        start, end = int(start), int(end)
        if not (0 <= start <= end < len(gt)):
            raise ValueError("Window outside annotated frame range")
        sums[start:end + 1] += prob
        counts[start:end + 1] += 1
    if np.any(counts == 0):
        raise ValueError("Some frames have no covering inference windows")
    return gt, np.argmax(sums / counts[:, None], axis=1).astype(np.int64)


def resolve_video(path_hint, root, video_id=None):
    hint = Path(str(path_hint or ""))
    if hint.is_file():
        return hint
    if root is None:
        return None
    root = Path(root)
    if hint.name:
        matches = list(root.rglob(hint.name))
        if matches:
            return matches[0]
    if video_id is not None:
        for ext in ("mp4", "avi", "mov"):
            matches = list(root.rglob(str(video_id) + "." + ext))
            if matches:
                return matches[0]
    return None


def make_sheet(video_path, start, end, fps, text_lines, output, is_ipn=False):
    """Representative frames at 10%, 50%, 90% of an annotated interval.

    These are diagnostic snapshots, NOT learned attention/keyframe scores.
    IPN annotation indexing is 1-based; CTP annotation is 0-based.
    """
    if video_path is None or not Path(video_path).is_file():
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    imgs = []
    for frac in (0.10, 0.50, 0.90):
        idx = int(round(start + frac * max(0, end - start)))
        seek = max(0, idx - (1 if is_ipn else 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, f"frame={idx} ({frac:.0%})", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        imgs.append(frame)
    cap.release()
    if not imgs:
        return False
    while len(imgs) < 3:
        imgs.append(np.zeros_like(imgs[0]))
    canvas = np.vstack([np.zeros((60, 960, 3), np.uint8),
                        np.hstack(imgs)])
    for k, line in enumerate(text_lines[:2]):
        cv2.putText(canvas, str(line)[:105], (8, 22 + k * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), canvas))


def isolated_analysis(csv, out, dataset, root, top):
    df = pd.read_csv(csv)
    mandatory = {"gt_idx", "pred_idx", "t_start", "t_end"}
    if not mandatory.issubset(df.columns):
        raise ValueError("Missing columns in isolated CSV: " +
                         repr(sorted(mandatory - set(df.columns))))
    y = df.gt_idx.astype(int).to_numpy()
    pred = df.pred_idx.astype(int).to_numpy()
    save_json(out / "isolated_metrics.json", metric_pack(y, pred))
    pd.DataFrame(classification_report(y, pred, output_dict=True,
                                        zero_division=0)).T.to_csv(
        out / "isolated_classification_report.csv")
    labels = sorted(set(y) | set(pred))
    pd.DataFrame(confusion_matrix(y, pred, labels=labels),
                 index=labels, columns=labels).to_csv(
        out / "isolated_confusion_matrix.csv")
    if dataset == "ctp":
        mask = y != 0
        save_json(out / "isolated_gesture_only_metrics.json",
                  metric_pack(y[mask], pred[mask], labels=list(range(1, 9))))
    errors = df.loc[y != pred].copy()
    errors["error_type"] = errors.apply(
        lambda r: f'GT_{int(r.gt_idx)}_PRED_{int(r.pred_idx)}', axis=1)
    # Highest confidence errors are especially useful for inspecting failure cases.
    errors = errors.sort_values("confidence", ascending=False) if "confidence" in errors else errors
    errors.to_csv(out / "isolated_misclassified_segments.csv", index=False)
    pics, log = [], []
    for rank, (_, r) in enumerate(errors.head(top).iterrows(), 1):
        video_hint = r.get("video_path", "")
        vid_id = r.get("video_id", r.get("video", ""))
        video = resolve_video(video_hint, root, vid_id)
        filename = f"error_{rank:03d}_gt{int(r.gt_idx)}_pred{int(r.pred_idx)}.png"
        dest = out / "keyframe_error_montages" / filename
        ok = make_sheet(
            video, int(r.t_start), int(r.t_end), 0,
            [f"{vid_id} GT={int(r.gt_idx)} PRED={int(r.pred_idx)}",
             f"confidence={r.get('confidence', float('nan'))}"],
            dest, is_ipn=(dataset == "ipn"))
        if ok:
            pics.append(str(dest))
        log.append(f"rank={rank} video={vid_id} frame=[{int(r.t_start)},{int(r.t_end)}] "
                   f"GT={int(r.gt_idx)} pred={int(r.pred_idx)} "
                   f"confidence={r.get('confidence','?')} "
                   f"visual={dest if ok else 'VIDEO_NOT_FOUND'}")
    (out / "error_keyframes.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    return {"evaluated_clips": len(df), "wrong_clips": len(errors),
            "rendered_contact_sheets": len(pics)}


def continuous_analysis(path, out, top, root, dataset):
    with Path(path).open("rb") as fh:
        caches = pickle.load(fh)  # Trusted cache created by the user's notebook only.
    if not isinstance(caches, dict):
        raise ValueError("Expected per-video dictionary raw cache")
    per = []
    all_gt, all_pred = [], []
    fail_segments = []
    for name, entry in sorted(caches.items()):
        gt, pred = frame_probs_from_cache(entry)
        fps = float(entry.get("fps", 15))
        result = {"video": str(name), "fps": fps,
                  **metric_pack(gt, pred)}
        gt_seq = [v["label"] for v in segments_from_labels(gt)]
        pr_seq = [v["label"] for v in segments_from_labels(pred)]
        dist = edit_distance(gt_seq, pr_seq)
        result.update(gt_gestures=len(gt_seq), predicted_gestures=len(pr_seq),
                      edit_distance=dist,
                      levenshtein_gt_normalized_pct=100 * (1 - dist / max(1, len(gt_seq))))
        per.append(result)
        all_gt.append(gt)
        all_pred.append(pred)
        errors = segments_from_labels((gt != pred).astype(np.int64), ignore_zero=True)
        for seg in errors:
            if seg["end"] - seg["start"] + 1 < max(3, int(round(fps * 0.3))):
                continue
            sample = (seg["start"] + seg["end"]) // 2
            fail_segments.append({
                "video": str(name), "video_path": entry.get("video_path", ""),
                "start": seg["start"], "end": seg["end"], "fps": fps,
                "gt_at_mid": int(gt[sample]), "pred_at_mid": int(pred[sample]),
                "wrong_frames": seg["end"] - seg["start"] + 1
            })
    df = pd.DataFrame(per)
    df.to_csv(out / "continuous_raw_per_video_extra_metrics.csv", index=False)
    summary = metric_pack(np.concatenate(all_gt), np.concatenate(all_pred))
    summary["mean_per_video_frame_accuracy"] = float(df["accuracy"].mean())
    summary["mean_per_video_macro_f1"] = float(df["macro_f1"].mean())
    summary["mean_per_video_edit_score_gt_pct_reference_only"] = float(
        df["levenshtein_gt_normalized_pct"].mean())
    summary["note"] = (
        "Offline, noncausal full-window aggregation. Levenshtein formula shown is "
        "a diagnostic reference only; verify exact IPN official metric protocol.")
    save_json(out / "continuous_raw_metrics_extra.json", summary)
    errs = pd.DataFrame(fail_segments)
    if errs.empty:
        return {"videos": len(df), "hard_frame_segments": 0, "rendered_contact_sheets": 0}
    errs = errs.sort_values("wrong_frames", ascending=False)
    errs.to_csv(out / "continuous_hard_frame_segments.csv", index=False)
    log = []
    successes = 0
    for rank, row in enumerate(errs.head(top).itertuples(), 1):
        video = resolve_video(row.video_path, root, row.video)
        dest = out / "keyframe_error_montages_continuous" / f"error_{rank:03d}.png"
        ok = make_sheet(video, row.start, row.end, row.fps,
                        [f"video={row.video} GT={row.gt_at_mid} pred={row.pred_at_mid}",
                         f"bad_frames={row.wrong_frames}"], dest, is_ipn=False)
        successes += int(ok)
        log.append(f"rank={rank} video={row.video} GT={row.gt_at_mid} "
                   f"PRED={row.pred_at_mid} start={row.start} end={row.end} "
                   f"visual={dest if ok else 'VIDEO_NOT_FOUND'}")
    (out / "continuous_error_keyframes.txt").write_text(
        "\n".join(log) + "\n", encoding="utf-8")
    return {"videos": len(df), "hard_frame_segments": len(errs),
            "rendered_contact_sheets": successes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ipn", "ctp"], required=True)
    parser.add_argument("--isolated", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top", type=int, default=30)
    a = parser.parse_args()
    if not a.isolated and not a.cache:
        parser.error("Provide --isolated and/or --cache")
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": a.dataset}
    if a.isolated:
        summary["isolated"] = isolated_analysis(
            a.isolated, a.out, a.dataset, a.video_root, a.top)
    if a.cache:
        summary["continuous"] = continuous_analysis(
            a.cache, a.out, a.top, a.video_root, a.dataset)
    save_json(a.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved outputs to", a.out)


if __name__ == "__main__":
    main()
