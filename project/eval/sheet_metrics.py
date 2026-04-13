"""
Sheet-level correctness metrics.
"""

from __future__ import annotations

from collections import defaultdict


def compute_sheet_metrics(records: list[dict]) -> dict:
    by_sheet: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        sheet_id = record.get("sheet_id")
        if sheet_id:
            by_sheet[sheet_id].append(record)

    if not by_sheet:
        return {
            "n_sheets": 0,
            "sheet_coverage_mean": None,
            "sheet_panel_accuracy_mean": None,
            "sheet_copy_violation_rate_mean": None,
        }

    coverage_scores = []
    panel_accuracy_scores = []
    copy_violation_scores = []

    for sheet_records in by_sheet.values():
        requested = {
            r["requested_label"]
            for r in sheet_records
            if r.get("requested_label") not in (None, "unknown")
        }
        correctly_predicted = {
            r["requested_label"]
            for r in sheet_records
            if r.get("requested_label") not in (None, "unknown")
            and r.get("predicted_label") == r.get("requested_label")
        }

        if requested:
            coverage_scores.append(len(correctly_predicted) / len(requested))

        labeled_records = [
            r for r in sheet_records if r.get("requested_label") not in (None, "unknown")
        ]
        if labeled_records:
            panel_hits = sum(
                1.0
                for r in labeled_records
                if r.get("predicted_label") == r.get("requested_label")
            )
            panel_accuracy_scores.append(panel_hits / len(labeled_records))

        violations = [
            1.0 if r.get("copy_violation") else 0.0
            for r in sheet_records
            if r.get("copy_score") is not None
        ]
        if violations:
            copy_violation_scores.append(sum(violations) / len(violations))

    return {
        "n_sheets": len(by_sheet),
        "sheet_coverage_mean": (
            sum(coverage_scores) / len(coverage_scores) if coverage_scores else None
        ),
        "sheet_panel_accuracy_mean": (
            sum(panel_accuracy_scores) / len(panel_accuracy_scores) if panel_accuracy_scores else None
        ),
        "sheet_copy_violation_rate_mean": (
            sum(copy_violation_scores) / len(copy_violation_scores) if copy_violation_scores else None
        ),
    }
