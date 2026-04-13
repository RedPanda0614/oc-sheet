"""
Proposal-aligned evaluation runner.

Metrics covered:
- identity preservation (ArcFace cosine similarity)
- palette consistency (Lab-space palette distance proxy)
- image quality (FID, optional if clean-fid is installed)
- expression / view control accuracy
- copy score
- sheet-level correctness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcface_similarity import ArcFaceEvaluator
from copy_score import copy_score, copy_violation
from expression_classifier import CLIPControlEvaluator
from fid_score import compute_fid
from palette_distance import palette_distance
from sheet_metrics import compute_sheet_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", default="data/pairs/val.json")
    parser.add_argument("--generated-dir", default="results/baseline/batch")
    parser.add_argument("--manifest", default=None, help="Optional explicit manifest path")
    parser.add_argument("--output-json", default="results/metrics_proposal_eval.json")
    parser.add_argument("--copy-threshold", type=float, default=0.88)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-control", action="store_true")
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


def load_manifest(manifest_path: str | None, pairs: list[dict], generated_dir: str) -> list[dict]:
    gen_dir = Path(resolve_path(generated_dir))
    if manifest_path:
        manifest = Path(resolve_path(manifest_path))
    else:
        manifest = gen_dir / "manifest.json"

    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if isinstance(payload, dict) and "records" in payload:
            return payload["records"]
        return payload

    records = []
    for idx, pair in enumerate(pairs):
        prefix = f"{idx:04d}_"
        matches = sorted(gen_dir.glob(f"{prefix}*.jpg")) + sorted(gen_dir.glob(f"{prefix}*.png"))
        if not matches:
            continue

        generated_path = matches[0]
        stem_parts = generated_path.stem.split("_")
        requested_label = stem_parts[-1] if len(stem_parts) >= 2 else pair.get("target_emotion", "unknown")
        records.append(
            {
                "index": idx,
                "sheet_id": pair.get("sheet_id"),
                "reference_path": resolve_path(pair.get("reference_path")),
                "target_path": resolve_path(pair.get("target_path")),
                "generated_path": str(generated_path.resolve()),
                "requested_label": requested_label,
                "label_type": "expression",
            }
        )
    return records


def mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def evaluate_records(records: list[dict], args) -> tuple[list[dict], dict]:
    identity_eval = ArcFaceEvaluator()
    control_eval = None if args.skip_control else CLIPControlEvaluator()

    real_paths = []
    fake_paths = []

    identity_scores = []
    palette_scores = []
    copy_scores = []
    expression_hits = []
    control_confidences = []
    view_hits = []

    for record in records:
        generated_path = record["generated_path"]
        reference_path = record["reference_path"]
        target_path = record.get("target_path")
        requested_label = record.get("requested_label", "unknown")
        label_type = record.get("label_type", "expression")

        identity = None
        palette = None
        copied = None
        copy_flag = None
        predicted = None
        pred_conf = None

        if reference_path and Path(reference_path).exists() and Path(generated_path).exists():
            identity = identity_eval.similarity(reference_path, generated_path)
            palette = palette_distance(reference_path, generated_path)
            copied = copy_score(reference_path, generated_path)
            copy_flag = copy_violation(copied, threshold=args.copy_threshold)

        if control_eval is not None and Path(generated_path).exists():
            pred = control_eval.predict(generated_path, label_type=label_type)
            predicted = pred.label
            pred_conf = pred.confidence
            if requested_label not in (None, "unknown"):
                hit = float(predicted == requested_label)
                if label_type == "expression":
                    expression_hits.append(hit)
                elif label_type == "view":
                    view_hits.append(hit)
            control_confidences.append(pred_conf)

        if target_path and Path(target_path).exists() and Path(generated_path).exists():
            real_paths.append(target_path)
            fake_paths.append(generated_path)

        record["identity_similarity"] = identity
        record["palette_distance"] = palette
        record["copy_score"] = copied
        record["copy_violation"] = copy_flag
        record["predicted_label"] = predicted
        record["prediction_confidence"] = pred_conf

        identity_scores.append(identity)
        palette_scores.append(palette)
        copy_scores.append(copied)

    copy_flags = [r.get("copy_violation") for r in records if r.get("copy_violation") is not None]

    metrics = {
        "n_evaluated": len(records),
        "identity_similarity_mean": mean_or_none(identity_scores),
        "palette_distance_mean": mean_or_none(palette_scores),
        "copy_score_mean": mean_or_none(copy_scores),
        "copy_violation_rate": (
            sum(1.0 for flag in copy_flags if flag) / len(copy_flags) if copy_flags else None
        ),
        "expression_accuracy": (
            sum(expression_hits) / len(expression_hits) if expression_hits else None
        ),
        "view_accuracy": (sum(view_hits) / len(view_hits) if view_hits else None),
        "control_confidence_mean": (
            sum(control_confidences) / len(control_confidences) if control_confidences else None
        ),
    }

    metrics["fid"] = None if args.skip_fid else compute_fid(real_paths, fake_paths)
    metrics.update(compute_sheet_metrics(records))
    return records, metrics


def main():
    args = parse_args()
    pairs = load_pairs(args.pairs_json)
    records = load_manifest(args.manifest, pairs, args.generated_dir)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    evaluated_records, metrics = evaluate_records(records, args)

    output_path = Path(resolve_path(args.output_json))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "records": evaluated_records,
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved detailed evaluation to {output_path}")


if __name__ == "__main__":
    main()
