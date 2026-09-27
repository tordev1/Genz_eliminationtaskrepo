"""Starter-kit metric (per the challenge spec): format check (--validate-only)
plus Score_A, Score_B and the combined model score M, computed exactly as
described in the challenge's "Evaluation & scoring" section.

Usage:
    python evaluate.py --pred predictions.json --validate-only
    python evaluate.py --pred predictions.json --gt ground_truth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]
THRESHOLDS = (0.3, 0.5, 0.7)
H_SEC = 5.0   # accident-anticipation horizon
W_SEC = 10.0  # alarm-matching window
THETA = 0.5   # alarm threshold
ALARM_MERGE_GAP = 2.0


class ValidationError(Exception):
    pass


def temporal_iou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def validate_predictions(pred: dict, gt_video_ids: List[str] = None) -> None:
    if "team" not in pred or not isinstance(pred["team"], str) or not pred["team"]:
        raise ValidationError("predictions.json missing a non-empty 'team' string")
    if "videos" not in pred or not isinstance(pred["videos"], dict):
        raise ValidationError("predictions.json missing a 'videos' object")

    if gt_video_ids is not None:
        missing = set(gt_video_ids) - set(pred["videos"].keys())
        if missing:
            raise ValidationError(f"predictions.json is missing videos: {sorted(missing)}")

    for vid, entry in pred["videos"].items():
        if "events" not in entry or not isinstance(entry["events"], list):
            raise ValidationError(f"[{vid}] missing 'events' list")
        if "risk" not in entry or not isinstance(entry["risk"], list):
            raise ValidationError(f"[{vid}] missing 'risk' list")

        by_class_intervals: Dict[str, List[Tuple[float, float]]] = {}
        for ev in entry["events"]:
            if not (isinstance(ev, list) and len(ev) == 3):
                raise ValidationError(f"[{vid}] malformed event (expected [start, end, label]): {ev}")
            start, end, label = ev
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValidationError(f"[{vid}] non-numeric event bounds: {ev}")
            if label not in CLASSES:
                raise ValidationError(f"[{vid}] unknown label {label!r} in event: {ev}")
            if not (start < end):
                raise ValidationError(f"[{vid}] start_sec >= end_sec in event: {ev}")
            for s, e in by_class_intervals.get(label, []):
                if start < e and s < end:
                    raise ValidationError(
                        f"[{vid}] overlapping segments of class {label!r}: {ev} overlaps ({s}, {e})"
                    )
            by_class_intervals.setdefault(label, []).append((start, end))

        prev_t = None
        for pair in entry["risk"]:
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValidationError(f"[{vid}] malformed risk sample (expected [t_sec, score]): {pair}")
            t_sec, score = pair
            if not isinstance(t_sec, (int, float)) or not isinstance(score, (int, float)):
                raise ValidationError(f"[{vid}] non-numeric risk sample: {pair}")
            if not (0.0 <= score <= 1.0):
                raise ValidationError(f"[{vid}] risk score out of [0, 1]: {pair}")
            if prev_t is not None and t_sec < prev_t:
                raise ValidationError(f"[{vid}] risk samples not in increasing time order near {pair}")
            prev_t = t_sec


# ---------------------------------------------------------------------------
# Score A -- event detection
# ---------------------------------------------------------------------------

def _match_class_in_video(gt_segs, pred_segs, tau):
    pairs = []
    for gi, (gs, ge) in enumerate(gt_segs):
        for pi, (ps, pe) in enumerate(pred_segs):
            i = temporal_iou(gs, ge, ps, pe)
            if i > 0:
                pairs.append((i, gi, pi))
    pairs.sort(key=lambda x: -x[0])

    matched_gt, matched_pred = set(), set()
    for i, gi, pi in pairs:
        if i < tau:
            break
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)

    tp = len(matched_gt)
    fp = len(pred_segs) - len(matched_pred)
    fn = len(gt_segs) - len(matched_gt)
    return tp, fp, fn


def compute_score_a(gt: dict, pred: dict):
    classes_present = set()
    for entry in gt.values():
        for _, _, c in entry["events"]:
            classes_present.add(c)
    for entry in pred["videos"].values():
        for _, _, c in entry["events"]:
            classes_present.add(c)

    per_class = {}
    per_class_per_tau = {}
    for c in sorted(classes_present):
        f1s = []
        for tau in THRESHOLDS:
            tp_total = fp_total = fn_total = 0
            for vid, gt_entry in gt.items():
                gt_segs = [(s, e) for s, e, cc in gt_entry["events"] if cc == c]
                pred_entry = pred["videos"].get(vid, {"events": []})
                pred_segs = [(s, e) for s, e, cc in pred_entry["events"] if cc == c]
                tp, fp, fn = _match_class_in_video(gt_segs, pred_segs, tau)
                tp_total += tp
                fp_total += fp
                fn_total += fn
            denom = 2 * tp_total + fp_total + fn_total
            f1 = (2 * tp_total / denom) if denom > 0 else 1.0
            f1s.append(f1)
        per_class_per_tau[c] = dict(zip(THRESHOLDS, f1s))
        per_class[c] = sum(f1s) / len(f1s)

    score_a = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return score_a, per_class, per_class_per_tau


# ---------------------------------------------------------------------------
# Score B -- accident anticipation
# ---------------------------------------------------------------------------

def _average_precision(labels: List[int], scores: List[float]) -> float:
    """sklearn-equivalent AP: sum_n (R_n - R_{n-1}) * P_n over rank-sorted
    scores (descending), no external dependency."""
    n_pos = sum(labels)
    if n_pos == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def _frame_labels_for_video(events, risk_times):
    """-1 = ignored, 1 = positive, 0 = negative, aligned to risk_times."""
    accidents = [(s, e) for s, e, c in events if c == "accident"]
    near_misses = [(s, e) for s, e, c in events if c == "near_miss"]

    labels = [0] * len(risk_times)
    for i, t in enumerate(risk_times):
        label = 0
        for s, e in accidents:
            if s - H_SEC <= t < s:
                label = 1
            if s <= t <= e:
                label = -1
                break
        if label != -1:
            for s, e in near_misses:
                if s - H_SEC <= t <= e:
                    label = -1
                    break
        labels[i] = label
    return labels


def _alarms_from_risk(risk):
    """Maximal runs with score >= THETA, merged across gaps < ALARM_MERGE_GAP."""
    if not risk:
        return []
    runs = []
    run_start = None
    run_end = None
    for t, score in risk:
        if score >= THETA:
            if run_start is None:
                run_start = t
            run_end = t
        else:
            if run_start is not None:
                runs.append([run_start, run_end])
                run_start = None
    if run_start is not None:
        runs.append([run_start, run_end])

    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] < ALARM_MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def compute_score_b(gt: dict, pred: dict):
    all_labels, all_scores = [], []
    total_matched_alarms = total_alarms = 0
    total_matched_accidents = total_accidents = 0
    ttas = []

    for vid, gt_entry in gt.items():
        events = gt_entry["events"]
        risk = pred["videos"].get(vid, {"risk": []})["risk"]
        risk_times = [r[0] for r in risk]
        risk_scores = [r[1] for r in risk]

        frame_labels = _frame_labels_for_video(events, risk_times)
        ignored_intervals = []  # for discarding alarms starting inside ignored frames
        for i, lbl in enumerate(frame_labels):
            if lbl == -1:
                ignored_intervals.append(risk_times[i])
            else:
                all_labels.append(lbl)
                all_scores.append(risk_scores[i])

        accidents = sorted([(s, e) for s, e, c in events if c == "accident"], key=lambda x: x[0])
        alarms = _alarms_from_risk(risk)

        def starts_in_ignored(t):
            return any(abs(t - it) < 1e-6 for it in ignored_intervals)

        alarms = [a for a in alarms if not starts_in_ignored(a[0])]
        total_alarms += len(alarms)
        total_accidents += len(accidents)

        matched_accident_idx = set()
        for a_start, _a_end in alarms:
            candidates = [
                (s, i) for i, (s, e) in enumerate(accidents)
                if i not in matched_accident_idx and (s - W_SEC) <= a_start < s
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])
            s, i = candidates[0]
            matched_accident_idx.add(i)
            total_matched_alarms += 1
            total_matched_accidents += 1
            ttas.append(s - a_start)

        for i in range(len(accidents)):
            if i not in matched_accident_idx:
                ttas.append(0.0)

    ap_raw = _average_precision(all_labels, all_scores)
    r = (sum(all_labels) / len(all_labels)) if all_labels else 0.0
    ap = max(0.0, (ap_raw - r) / (1 - r)) if r < 1 else 0.0

    precision = total_matched_alarms / total_alarms if total_alarms > 0 else 0.0
    recall = total_matched_accidents / total_accidents if total_accidents > 0 else 0.0
    f1_alarm = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    m_tta = sum(ttas) / len(ttas) if ttas else 0.0
    score_b = 0.4 * ap + 0.4 * f1_alarm + 0.2 * (m_tta / W_SEC)

    return score_b, {
        "AP_raw": ap_raw, "AP": ap, "precision": precision, "recall": recall,
        "F1_alarm": f1_alarm, "mTTA": m_tta, "total_accidents": total_accidents,
        "total_alarms": total_alarms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    with open(args.pred, "r", encoding="utf-8") as f:
        pred = json.load(f)

    gt = None
    if args.gt:
        with open(args.gt, "r", encoding="utf-8") as f:
            gt = json.load(f)

    try:
        validate_predictions(pred, gt_video_ids=list(gt.keys()) if gt else None)
    except ValidationError as e:
        print(f"INVALID: {e}")
        sys.exit(1)
    print("Format OK.")

    if args.validate_only or gt is None:
        return

    score_a, per_class, per_class_per_tau = compute_score_a(gt, pred)
    score_b, b_details = compute_score_b(gt, pred)
    model_score = 0.7 * score_a + 0.3 * score_b

    print("\nPer-class F1 (diagnostic, mean over tau 0.3/0.5/0.7):")
    for c in sorted(per_class):
        taus = per_class_per_tau[c]
        print(
            f"  {c:22s} mean={per_class[c]:.3f}  "
            f"(0.3={taus[0.3]:.3f} 0.5={taus[0.5]:.3f} 0.7={taus[0.7]:.3f})"
        )

    print(f"\nScore_A (event detection, macro F1): {score_a:.4f}")
    print(f"Score_B (accident anticipation):     {score_b:.4f}")
    for k, v in b_details.items():
        print(f"  {k}: {v}")
    print(f"\nModel score M = 0.7*A + 0.3*B:       {model_score:.4f}")


if __name__ == "__main__":
    main()
