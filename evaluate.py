#!/usr/bin/env python3
"""
evaluate.py — the official metric. Do not modify.

    python evaluate.py --pred predictions.json --gt ground_truth.json [--json report.json] [--per-video]
    python evaluate.py --pred predictions.json --validate-only            # format check without labels

Format check (runs first; any error -> exit code 1, nothing is scored)
    * top level {"videos": {<file name>: {"events": [...], "risk": [...]?}}}
    * event = [start_sec, end_sec, label]; numbers; 0 <= start < end; label in OFFICIAL_CLASSES
    * segments of the same class in one video do not overlap
    * risk = [[t_sec, score], ...]; t non-decreasing; 0 <= score <= 1
    * every ground-truth video is present (even with "events": []); end_sec <= duration + 0.5 s

Part A  (event detection)
    For each class c and each tIoU threshold tau in {0.3, 0.5, 0.7}: greedy
    one-to-one matching of predicted and ground-truth segments by descending
    temporal IoU (pairs below tau never match). TP/FP/FN are pooled over all
    videos, then F1_c(tau).  Score_A = mean over classes of mean over tau of F1_c(tau).
    Classes = those present in the ground truth OR in the predictions.

Part B  (accident anticipation; `accident` events only; H=5 s, W=10 s, theta=0.5)
    Frame labels: positive if s-H <= t < s for some accident starting at s;
    ignored inside any accident [s, e] and inside [s-H, e] of any near_miss;
    negative otherwise.
      AP        average precision of the risk score over labelled frames (pooled),
                chance-normalised: AP = max(0, (AP_raw - r) / (1 - r)), r = positive rate,
                so a constant or random score gets 0
      F1_alarm  alarms = maximal runs with score >= theta, runs closer than 2 s
                merged, alarm time = run start; an alarm in [s-W, s) of a
                still-unmatched accident matches it (earliest alarm wins);
                alarms starting on ignored frames are dropped
      mTTA      mean over accidents of (s - matched alarm start), 0 if unmatched
    Score_B = 0.4*AP + 0.4*F1_alarm + 0.2*mTTA/W

Model score  M = 0.7*Score_A + 0.3*Score_B   (M = Score_A if the test set has no accidents)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

OFFICIAL_CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

TIOU_THRESHOLDS = (0.3, 0.5, 0.7)
H = 5.0            # anticipation horizon (s)
W = 10.0           # alarm matching window before an accident (s)
THETA = 0.5        # alarm threshold on the risk score
MERGE_GAP = 2.0    # alarm runs closer than this are merged (s)
WEIGHT_A, WEIGHT_B = 0.7, 0.3
B_WEIGHTS = {"ap": 0.4, "f1_alarm": 0.4, "mtta": 0.2}


# ----------------------------------------------------------------------------
# format check
# ----------------------------------------------------------------------------
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate(pred, gt: dict | None = None, classes=OFFICIAL_CLASSES) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Any error means the file cannot be scored."""
    errors, warnings = [], []
    if not isinstance(pred, dict) or not isinstance(pred.get("videos"), dict):
        return ['top level must be {"videos": {...}}'], warnings
    if "team" not in pred:
        warnings.append('missing "team" name')

    for vid, entry in pred["videos"].items():
        where = f"videos[{vid!r}]"
        if "/" in vid or "\\" in vid:
            errors.append(f"{where}: key must be a file name, not a path")
        if not isinstance(entry, dict) or not isinstance(entry.get("events"), list):
            errors.append(f'{where}: must be {{"events": [...], "risk": [...]}}')
            continue
        by_class: dict[str, list[tuple[float, float]]] = {}
        for i, ev in enumerate(entry["events"]):
            w = f"{where}.events[{i}]"
            if not (isinstance(ev, list) and len(ev) == 3):
                errors.append(f"{w}: must be [start_sec, end_sec, label], got {ev!r}")
                continue
            s, e, label = ev
            if not (_is_num(s) and _is_num(e)):
                errors.append(f"{w}: start/end must be numbers, got {s!r}, {e!r}")
                continue
            if not (0 <= s < e):
                errors.append(f"{w}: need 0 <= start < end, got [{s}, {e}]")
            if not isinstance(label, str) or label not in classes:
                errors.append(f"{w}: label {label!r} not in the official class list")
                continue
            if gt is not None and vid in gt and e > float(gt[vid].get("duration", float("inf"))) + 0.5:
                errors.append(f"{w}: end {e} > video duration {gt[vid]['duration']}")
            by_class.setdefault(label, []).append((float(s), float(e)))
        for label, ss in by_class.items():
            ss.sort()
            for (s1, e1), (s2, e2) in zip(ss, ss[1:]):
                if s2 < e1:
                    errors.append(f"{where}: overlapping {label!r} segments [{s1}, {e1}] and [{s2}, {e2}]")
        risk = entry.get("risk", [])
        if not isinstance(risk, list):
            errors.append(f"{where}.risk: must be a list")
        else:
            prev_t = -1.0
            for j, item in enumerate(risk):
                if not (isinstance(item, list) and len(item) == 2 and _is_num(item[0]) and _is_num(item[1])):
                    errors.append(f"{where}.risk[{j}]: must be [t_sec, score], got {item!r}")
                    break
                if item[0] < prev_t:
                    errors.append(f"{where}.risk[{j}]: timestamps must be non-decreasing")
                    break
                if not (0.0 <= item[1] <= 1.0):
                    errors.append(f"{where}.risk[{j}]: score {item[1]} outside [0, 1]")
                    break
                prev_t = item[0]
            if not risk:
                warnings.append(f"{where}: no risk curve (Part B scores 0 for this video)")
        if gt is not None and vid not in gt:
            warnings.append(f"{where}: not in the ground truth, ignored")
    if gt is not None:
        for v in gt:
            if v not in pred["videos"]:
                errors.append(f"ground-truth video {v!r} missing from predictions (must be present, even with [])")
    return errors, warnings


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def tiou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def match_segments(gt: list[tuple[float, float]], pred: list[tuple[float, float]], thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching by descending IoU. Returns (tp, fp, fn)."""
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            iou = tiou(g, p)
            if iou >= thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_g, used_p = set(), set()
    for _, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
    tp = len(used_g)
    return tp, len(pred) - tp, len(gt) - tp


def segs(events, label: str | None = None) -> list[tuple[float, float]]:
    return [(float(s), float(e)) for s, e, lab in events if label is None or lab == label]


# ----------------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------------
def evaluate_part_a(gt: dict, pred_videos: dict, per_video: bool = False) -> dict:
    classes = sorted(
        {lab for v in gt.values() for _, _, lab in v["events"]}
        | {lab for v in pred_videos.values() for _, _, lab in v.get("events", [])}
    )
    counts = {c: {t: [0, 0, 0] for t in TIOU_THRESHOLDS} for c in classes}
    micro = {t: [0, 0, 0] for t in TIOU_THRESHOLDS}
    agnostic = {t: [0, 0, 0] for t in TIOU_THRESHOLDS}
    video_rows = []

    for vid, g in gt.items():
        p_events = pred_videos.get(vid, {}).get("events", [])
        row = {"video": vid, "gt": len(g["events"]), "pred": len(p_events)}
        for c in classes:
            gs, ps = segs(g["events"], c), segs(p_events, c)
            for t in TIOU_THRESHOLDS:
                tp, fp, fn = match_segments(gs, ps, t)
                for k, val in enumerate((tp, fp, fn)):
                    counts[c][t][k] += val
                    micro[t][k] += val
                if t == 0.5:
                    row[c] = f"{tp}/{fp}/{fn}"
        for t in TIOU_THRESHOLDS:
            tp, fp, fn = match_segments(segs(g["events"]), segs(p_events), t)
            for k, val in enumerate((tp, fp, fn)):
                agnostic[t][k] += val
        video_rows.append(row)

    per_class = {}
    for c in classes:
        per_class[c] = {str(t): prf(*counts[c][t]) for t in TIOU_THRESHOLDS}
        per_class[c]["f1_mean"] = sum(per_class[c][str(t)]["f1"] for t in TIOU_THRESHOLDS) / len(TIOU_THRESHOLDS)
    score_a = sum(per_class[c]["f1_mean"] for c in classes) / len(classes) if classes else 0.0

    return {
        "score_a": score_a,
        "classes": classes,
        "per_class": per_class,
        "micro": {str(t): prf(*micro[t]) for t in TIOU_THRESHOLDS},
        "class_agnostic": {str(t): prf(*agnostic[t]) for t in TIOU_THRESHOLDS},
        "per_video": video_rows if per_video else None,
    }


# ----------------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------------
def frame_label(t: float, accidents: list[tuple[float, float]], near_misses: list[tuple[float, float]]):
    """+1 positive, 0 negative, None ignored."""
    for s, e in accidents:
        if s <= t <= e:
            return None
    for s, e in accidents:
        if s - H <= t < s:
            return 1
    for s, e in near_misses:
        if s - H <= t <= e:
            return None
    return 0


def average_precision(scores: list[float], labels: list[int]) -> float:
    """AP = sum_k (R_k - R_{k-1}) * P_k over unique score thresholds (sklearn definition)."""
    n_pos = sum(labels)
    if n_pos == 0 or not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ap, tp, fp, prev_recall, i = 0.0, 0, 0, 0.0, 0
    while i < len(order):
        thr = scores[order[i]]
        while i < len(order) and scores[order[i]] == thr:   # take the whole tie group
            if labels[order[i]]:
                tp += 1
            else:
                fp += 1
            i += 1
        precision, recall = tp / (tp + fp), tp / n_pos
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def alarm_starts(curve: list[list[float]], theta: float = THETA, merge_gap: float = MERGE_GAP) -> list[float]:
    runs, start, end = [], None, None
    for t, s in curve:
        if s >= theta:
            if start is None:
                start = t
            end = t
        elif start is not None:
            runs.append((start, end))
            start = None
    if start is not None:
        runs.append((start, end))
    merged: list[list[float]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [m[0] for m in merged]


def evaluate_part_b(gt: dict, pred_videos: dict) -> dict | None:
    n_accidents = sum(1 for v in gt.values() for _, _, lab in v["events"] if lab == "accident")
    if n_accidents == 0:
        return None

    all_scores, all_labels = [], []
    n_alarms, n_matched, tta_sum = 0, 0, 0.0
    per_video = {}

    for vid, g in gt.items():
        accidents = sorted(segs(g["events"], "accident"))
        near_misses = segs(g["events"], "near_miss")
        curve = pred_videos.get(vid, {}).get("risk", []) or []

        # frame-level AP samples
        for t, s in curve:
            lab = frame_label(float(t), accidents, near_misses)
            if lab is not None:
                all_scores.append(float(s))
                all_labels.append(lab)

        # alarms
        starts = [a for a in alarm_starts(curve) if frame_label(a, accidents, near_misses) is not None]
        n_alarms += len(starts)
        unmatched = list(starts)
        ttas = []
        for s, _ in accidents:
            cands = [a for a in unmatched if s - W <= a < s]
            if cands:
                a = min(cands)
                unmatched.remove(a)
                ttas.append(s - a)
                n_matched += 1
                tta_sum += s - a
            else:
                ttas.append(0.0)
        per_video[vid] = {"accidents": len(accidents), "alarms": len(starts), "tta": [round(x, 2) for x in ttas]}

    ap_raw = average_precision(all_scores, all_labels)
    pos_rate = sum(all_labels) / len(all_labels) if all_labels else 0.0
    # chance-normalised AP: a constant (or random) score gets 0, a perfect ranking gets 1
    ap = max(0.0, (ap_raw - pos_rate) / (1.0 - pos_rate)) if pos_rate < 1.0 else 0.0
    precision = n_matched / n_alarms if n_alarms else 0.0
    recall = n_matched / n_accidents
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mtta = tta_sum / n_accidents
    score_b = B_WEIGHTS["ap"] * ap + B_WEIGHTS["f1_alarm"] * f1 + B_WEIGHTS["mtta"] * min(1.0, mtta / W)
    return {
        "score_b": score_b,
        "ap": ap,
        "ap_raw": ap_raw,
        "positive_rate": pos_rate,
        "alarm_precision": precision,
        "alarm_recall": recall,
        "f1_alarm": f1,
        "mtta_sec": mtta,
        "n_accidents": n_accidents,
        "n_alarms": n_alarms,
        "n_matched": n_matched,
        "n_frames_scored": len(all_scores),
        "n_frames_positive": sum(all_labels),
        "per_video": per_video,
    }


# ----------------------------------------------------------------------------
def evaluate(gt: dict, pred: dict, per_video: bool = False) -> dict:
    pred_videos = pred.get("videos", {})
    missing = [v for v in gt if v not in pred_videos]
    extra = [v for v in pred_videos if v not in gt]
    a = evaluate_part_a(gt, pred_videos, per_video)
    b = evaluate_part_b(gt, pred_videos)
    model = WEIGHT_A * a["score_a"] + WEIGHT_B * b["score_b"] if b else a["score_a"]
    return {"model_score": model, "part_a": a, "part_b": b,
            "missing_videos": missing, "extra_videos": extra, "team": pred.get("team")}


def print_report(rep: dict) -> None:
    a, b = rep["part_a"], rep["part_b"]
    print(f"Team: {rep.get('team')}")
    if rep["missing_videos"]:
        print(f"  ! {len(rep['missing_videos'])} ground-truth video(s) missing from predictions (scored as empty): "
              + ", ".join(rep["missing_videos"]))
    if rep["extra_videos"]:
        print(f"  ! {len(rep['extra_videos'])} predicted video(s) not in ground truth (ignored)")

    print("\nPart A — event detection (TP/FP/FN pooled over videos)")
    hdr = f"{'class':<20}" + "".join(f"{'F1@' + str(t):>9}" for t in TIOU_THRESHOLDS) + f"{'mean':>9}{'TP/FP/FN@0.5':>16}"
    print(hdr)
    for c in a["classes"]:
        pc = a["per_class"][c]
        m = pc["0.5"]
        print(f"{c:<20}" + "".join(f"{pc[str(t)]['f1']:>9.3f}" for t in TIOU_THRESHOLDS)
              + f"{pc['f1_mean']:>9.3f}{str(m['tp']) + '/' + str(m['fp']) + '/' + str(m['fn']):>16}")
    print(f"{'micro (all events)':<20}" + "".join(f"{a['micro'][str(t)]['f1']:>9.3f}" for t in TIOU_THRESHOLDS))
    print(f"{'class-agnostic':<20}" + "".join(f"{a['class_agnostic'][str(t)]['f1']:>9.3f}" for t in TIOU_THRESHOLDS))
    print(f"Score A = {a['score_a']:.4f}")
    if a.get("per_video"):
        print("\n  per video (TP/FP/FN at tIoU 0.5):")
        for row in a["per_video"]:
            cells = ", ".join(f"{c}={row[c]}" for c in a["classes"] if row.get(c) and row[c] != "0/0/0")
            print(f"  {row['video']:<28} gt={row['gt']:<3} pred={row['pred']:<3} {cells}")

    print("\nPart B — accident anticipation")
    if b is None:
        print("  no accidents in the ground truth: Part B not scored, model score = Score A")
    else:
        print(f"  frames scored {b['n_frames_scored']} (positive {b['n_frames_positive']}), "
              f"accidents {b['n_accidents']}, alarms {b['n_alarms']}, matched {b['n_matched']}")
        print(f"  AP = {b['ap']:.3f} (raw {b['ap_raw']:.3f}, chance {b['positive_rate']:.3f})   "
              f"alarm P/R/F1 = {b['alarm_precision']:.3f}/{b['alarm_recall']:.3f}/{b['f1_alarm']:.3f}   "
              f"mTTA = {b['mtta_sec']:.2f} s")
        print(f"Score B = {b['score_b']:.4f}")
    print(f"\nMODEL SCORE = {rep['model_score']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", help="ground_truth.json (required unless --validate-only)")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--per-video", action="store_true")
    ap.add_argument("--validate-only", action="store_true", help="format check only, no scoring")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())
    gt = json.loads(Path(args.gt).read_text()) if args.gt else None
    if gt is None and not args.validate_only:
        ap.error("--gt is required unless --validate-only")

    errors, warnings = validate(pred, gt)
    for w in warnings:
        print("warning:", w)
    for e in errors:
        print("ERROR:", e)
    n_events = sum(len(v.get("events", [])) for v in pred.get("videos", {}).values()) if not errors else 0
    print(f"format: {len(pred.get('videos', {}))} video(s), {n_events} event(s), "
          f"{len(errors)} error(s), {len(warnings)} warning(s) -> {'VALID' if not errors else 'INVALID'}")
    if errors:
        return 1
    if args.validate_only:
        return 0

    print()
    rep = evaluate(gt, pred, per_video=args.per_video)
    print_report(rep)
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
        print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
