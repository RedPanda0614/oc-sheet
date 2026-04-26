"""
Evaluate whether generated images match the ground-truth target emotion.

Unlike run_eval.py, which can operate on the manifest's requested label, this
script explicitly looks up the labeled `target_emotion` from pairs.json and
measures generated-image accuracy against that ground truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from expression_classifier import CLIPControlEvaluator, EXPRESSION_PROMPTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/pairs/val.json")
    parser.add_argument("--generated-dir", default="results/baseline/batch")
    parser.add_argument("--manifest", default=None, help="Optional explicit manifest path")
    parser.add_argument("--output-json", default="results/metrics_target_emotion.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--exclude-expression-labels",
        default="",
        help="Comma-separated expression labels to remove from target labels and CLIP candidates.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((project_root() / path).resolve())


def load_pairs(pairs_json: str) -> list[dict]:
    return json.loads(Path(resolve_path(pairs_json)).read_text())


def load_manifest(manifest_path: str | None, generated_dir: str) -> list[dict]:
    gen_dir = Path(resolve_path(generated_dir))
    if manifest_path:
        manifest = Path(resolve_path(manifest_path))
    else:
        manifest = gen_dir / "manifest.json"

    if not manifest.exists():
        raise FileNotFoundError(
            f"Could not find manifest at {manifest}. Pass --manifest explicitly or generate one first."
        )
    payload = json.loads(manifest.read_text())
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    return payload


def build_pair_lookup(pairs: list[dict]) -> tuple[dict[int, dict], dict[str, dict]]:
    by_index = {idx: pair for idx, pair in enumerate(pairs)}
    by_target_path = {}
    for pair in pairs:
        target_path = pair.get("target_path")
        if target_path:
            by_target_path[resolve_path(target_path)] = pair
    return by_index, by_target_path


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_label_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main():
    args = parse_args()
    pairs = load_pairs(args.pairs_json)
    records = load_manifest(args.manifest, args.generated_dir)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    by_index, by_target_path = build_pair_lookup(pairs)
    excluded_expression_labels = parse_label_set(args.exclude_expression_labels)
    evaluator = CLIPControlEvaluator(exclude_expression_labels=excluded_expression_labels)

    detailed_records = []
    correct = 0
    confidences = []
    skipped_missing_gt = 0
    skipped_missing_image = 0

    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)

    valid_labels = set(EXPRESSION_PROMPTS.keys()) - excluded_expression_labels

    for record in records:
        generated_path = resolve_path(record.get("generated_path"))
        if not generated_path or not Path(generated_path).exists():
            skipped_missing_image += 1
            continue

        pair = None
        index = record.get("index")
        if isinstance(index, int) and index in by_index:
            pair = by_index[index]
        else:
            target_path = resolve_path(record.get("target_path"))
            if target_path:
                pair = by_target_path.get(target_path)

        target_emotion = pair.get("target_emotion") if pair else None
        if target_emotion not in valid_labels:
            skipped_missing_gt += 1
            continue

        pred = evaluator.predict(generated_path, label_type="expression")
        is_correct = pred.label == target_emotion

        per_class_total[target_emotion] += 1
        per_class_correct[target_emotion] += int(is_correct)
        confusion[target_emotion][pred.label] += 1
        correct += int(is_correct)
        confidences.append(pred.confidence)

        detailed_records.append(
            {
                "index": index,
                "sheet_id": record.get("sheet_id"),
                "generated_path": generated_path,
                "reference_path": resolve_path(record.get("reference_path")),
                "target_path": resolve_path(record.get("target_path")),
                "ground_truth_target_emotion": target_emotion,
                "predicted_emotion": pred.label,
                "prediction_confidence": pred.confidence,
                "correct": bool(is_correct),
            }
        )

    n_scored = len(detailed_records)
    per_class_accuracy = {
        label: (per_class_correct[label] / per_class_total[label]) if per_class_total[label] else None
        for label in sorted(per_class_total.keys())
    }
    macro_accuracy_values = [acc for acc in per_class_accuracy.values() if acc is not None]

    payload = {
        "metrics": {
            "n_manifest_records": len(records),
            "n_scored": n_scored,
            "skipped_missing_ground_truth": skipped_missing_gt,
            "skipped_missing_generated_image": skipped_missing_image,
            "ground_truth_target_emotion_accuracy": (correct / n_scored) if n_scored else None,
            "macro_target_emotion_accuracy": mean_or_none(macro_accuracy_values),
            "prediction_confidence_mean": mean_or_none(confidences),
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": {
                gt_label: dict(sorted(pred_counts.items()))
                for gt_label, pred_counts in sorted(confusion.items())
            },
        },
        "records": detailed_records,
    }

    output_path = Path(resolve_path(args.output_json))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload["metrics"], indent=2))
    print(f"\nSaved detailed target-emotion evaluation to {output_path}")


if __name__ == "__main__":
    main()
