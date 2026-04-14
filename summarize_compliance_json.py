#!/usr/bin/env python3
"""
Summarize IBCS compliance audit results from a JSON report (e.g. ibcs_compliance_report.json).

Usage:
  python summarize_compliance_json.py
  python summarize_compliance_json.py path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_rule_key(rule: Any) -> str:
    if not rule or not isinstance(rule, str):
        return "unknown"
    r = rule.strip()
    if "|" in r:
        r = r.split("|", maxsplit=1)[0].strip()
    return r or "unknown"


def load_report(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return [x for x in data if isinstance(x, dict)]


def summarize(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    errors = [r for r in rows if "error" in r]
    parsed = [r for r in rows if "error" not in r]

    err_by_type = Counter(str(r.get("raw_error_type", "unknown")) for r in errors)

    compliant_true = sum(1 for r in parsed if r.get("compliant") is True)
    compliant_false = sum(1 for r in parsed if r.get("compliant") is False)
    compliant_unclear = len(parsed) - compliant_true - compliant_false

    rule_counts: Counter[str] = Counter()
    confidences: list[float] = []
    low_conf = 0
    violation_rows = 0

    for r in parsed:
        violations = r.get("violations")
        if not isinstance(violations, list):
            continue
        for v in violations:
            if not isinstance(v, dict):
                continue
            violation_rows += 1
            rule_counts[_normalize_rule_key(v.get("rule"))] += 1
            c = _safe_float(v.get("confidence"))
            if c is not None:
                confidences.append(c)
            if v.get("low_confidence") is True:
                low_conf += 1

    chart_counts_per_file: list[int] = []
    chart_types: Counter[str] = Counter()
    starts_zero = Counter({"true": 0, "false": 0, "unknown": 0})

    for r in parsed:
        charts = r.get("charts_detected")
        if isinstance(charts, list):
            chart_counts_per_file.append(len(charts))
            for ch in charts:
                if not isinstance(ch, dict):
                    continue
                chart_types[str(ch.get("type") or "unknown")] += 1
                s = ch.get("starts_at_zero")
                if s is True:
                    starts_zero["true"] += 1
                elif s is False:
                    starts_zero["false"] += 1
                else:
                    starts_zero["unknown"] += 1
        else:
            chart_counts_per_file.append(0)

    def pct(n: int, d: int) -> str:
        if d == 0:
            return "n/a"
        return f"{100.0 * n / d:.1f}%"

    print("=== IBCS compliance report summary ===\n")
    print(f"Total entries:        {total}")
    print(f"  Parsed (no error): {len(parsed)}  ({pct(len(parsed), total)})")
    print(f"  Errors:             {len(errors)}  ({pct(len(errors), total)})")
    if err_by_type:
        print("\nErrors by type:")
        for k, v in err_by_type.most_common():
            print(f"  {k}: {v}")

    print("\n--- Compliance (parsed files only) ---")
    print(f"  Compliant: {compliant_true}  ({pct(compliant_true, len(parsed))})")
    print(f"  Non-compliant:  {compliant_false}  ({pct(compliant_false, len(parsed))})")
    print(f"  Unclear flag:   {compliant_unclear}  ({pct(compliant_unclear, len(parsed))})")

    print("\n--- Violations (aggregated) ---")
    print(f"  Total violation objects: {violation_rows}")
    print(f"  With low_confidence:     {low_conf}")
    if confidences:
        print(
            f"  Confidence: min={min(confidences):.2f} max={max(confidences):.2f} "
            f"mean={statistics.mean(confidences):.2f} "
            f"median={statistics.median(confidences):.2f}"
        )
        ge06 = sum(1 for c in confidences if c >= 0.6)
        print(f"  Violations with confidence >= 0.6: {ge06} / {len(confidences)}")
    if rule_counts:
        print("  By rule:")
        for rule, n in rule_counts.most_common():
            print(f"    {rule}: {n}")

    print("\n--- charts_detected ---")
    if chart_counts_per_file:
        print(
            f"  Charts per file: min={min(chart_counts_per_file)} "
            f"max={max(chart_counts_per_file)} "
            f"mean={statistics.mean(chart_counts_per_file):.2f}"
        )
    print(f"  starts_at_zero counts: {dict(starts_zero)}")
    if chart_types:
        print("  Types:")
        for t, n in chart_types.most_common(15):
            print(f"    {t}: {n}")
        if len(chart_types) > 15:
            print(f"    … and {len(chart_types) - 15} more type(s)")

    print("\n--- Pipeline performance ---")
    ok = len(parsed)
    print(
        f"  Successful structured audits: {ok}/{total} "
        f"({pct(ok, total)} of all files)"
    )
    if parsed:
        decided = compliant_true + compliant_false
        print(
            f"  Clear compliant/non-compliant: {decided}/{len(parsed)} "
            f"({pct(decided, len(parsed))} of parsed)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize IBCS compliance JSON report statistics."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        default=Path("ibcs_compliance_report.json"),
        help="Path to compliance report JSON (default: ./ibcs_compliance_report.json)",
    )
    args = parser.parse_args()
    path: Path = args.json_path
    if not path.is_file():
        raise SystemExit(f"File not found: {path.resolve()}")
    rows = load_report(path)
    summarize(rows)


if __name__ == "__main__":
    main()
