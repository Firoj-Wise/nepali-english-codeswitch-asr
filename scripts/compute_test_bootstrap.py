#!/usr/bin/env python3
"""Compute exact paired bootstrap confidence intervals on clean real_test.jsonl."""

import json
import logging
from pathlib import Path

from csrl.application.evaluate import score_pairs
from csrl.data.manifest import load_manifest
from csrl.domain.entities import ErrorCounts
from csrl.infrastructure.logging import configure_logging, get_logger
from csrl.metrics.bootstrap import paired_bootstrap_difference

_logger = get_logger(__name__)
MANIFEST = Path("data/splits/real_test.jsonl")

def load_hyps(run_id: str) -> list[str]:
    p = Path("experiments") / run_id / "hyps_test.jsonl"
    hyps = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                hyps.append(json.loads(line)["hypothesis"])
    return hyps

def main() -> None:
    configure_logging(logging.INFO, None)
    utterances = load_manifest(MANIFEST)
    _logger.info("Loaded %d test utterances.", len(utterances))

    s0_hyps = load_hyps("S0-20261021-real-test-clean")
    s1_hyps = load_hyps("S1-20261021-real-test-clean")
    s4_hyps = load_hyps("S4-20261021-real-test-clean")

    _logger.info("Scoring pairs...")
    s0_scores = score_pairs(utterances, s0_hyps)
    s1_scores = score_pairs(utterances, s1_hyps)
    s4_scores = score_pairs(utterances, s4_hyps)

    # Extract ErrorCounts per utterance for MER
    s0_mer = [s.mer for s in s0_scores]
    s1_mer = [s.mer for s in s1_scores]
    s4_mer = [s.mer for s in s4_scores]

    # For SPER:
    s0_sper = [s.sper for s in s0_scores]
    s1_sper = [s.sper for s in s1_scores]
    s4_sper = [s.sper for s in s4_scores]

    # For SCR (English):
    s0_scr_en = [ErrorCounts(s.confusion.confusions_en, 0, 0, s.confusion.eligible_en) for s in s0_scores]
    s1_scr_en = [ErrorCounts(s.confusion.confusions_en, 0, 0, s.confusion.eligible_en) for s in s1_scores]
    s4_scr_en = [ErrorCounts(s.confusion.confusions_en, 0, 0, s.confusion.eligible_en) for s in s4_scores]

    comparisons = [
        ("S4 vs S0", s4_mer, s0_mer, "MER"),
        ("S1 vs S0", s1_mer, s0_mer, "MER"),
        ("S4 vs S1", s4_mer, s1_mer, "MER"),
        ("S4 vs S0", s4_sper, s0_sper, "SPER"),
        ("S1 vs S0", s1_sper, s0_sper, "SPER"),
        ("S4 vs S1", s4_sper, s1_sper, "SPER"),
        ("S4 vs S0", s4_scr_en, s0_scr_en, "SCR_en"),
        ("S1 vs S0", s1_scr_en, s0_scr_en, "SCR_en"),
        ("S4 vs S1", s4_scr_en, s1_scr_en, "SCR_en"),
    ]

    _logger.info("PAIRED BOOTSTRAP CONFIDENCE INTERVALS (95%% CI, 1000 samples, seed=42)")
    results = {}
    for name, sys_a, sys_b, metric in comparisons:
        ci = paired_bootstrap_difference(sys_a, sys_b, seed=42)
        diff_pct = ci.point_estimate * 100.0
        low_pct = ci.lower * 100.0
        high_pct = ci.upper * 100.0
        results[(name, metric)] = (diff_pct, low_pct, high_pct)
        _logger.info("%s | %s | Diff: %+.2f%% | 95%% CI: [%+.2f%%, %+.2f%%]", name, metric, diff_pct, low_pct, high_pct)

    # Save to a json for make_figures.py
    out_path = Path("experiments/bootstrap_test_clean.json")
    serializable = {f"{k[0]}|{k[1]}": v for k, v in results.items()}
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    _logger.info("Saved bootstrap results to %s", out_path)

if __name__ == "__main__":
    main()
