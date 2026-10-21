#!/usr/bin/env python3
"""Inspect episode distribution across dev and test and compute clustered block bootstrap."""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

from csrl.application.evaluate import score_pairs
from csrl.data.manifest import load_manifest
from csrl.domain.entities import ErrorCounts
from csrl.infrastructure.logging import configure_logging, get_logger

_logger = get_logger(__name__)

def main():
    dev_path = Path("data/splits/real_dev.jsonl")
    test_path = Path("data/splits/real_test.jsonl")
    
    dev_eps = set()
    dev_spks = set()
    for line in open(dev_path):
        if line.strip():
            d = json.loads(line)
            dev_eps.add(d["utterance_id"].rsplit("_", 1)[0])
            dev_spks.add(d.get("speaker_id", ""))
            
    test_eps = set()
    test_spks = set()
    for line in open(test_path):
        if line.strip():
            d = json.loads(line)
            test_eps.add(d["utterance_id"].rsplit("_", 1)[0])
            test_spks.add(d.get("speaker_id", ""))
            
    _logger.info("Dev episodes: %d (%s...)", len(dev_eps), sorted(list(dev_eps))[:3])
    _logger.info("Test episodes: %d (%s...)", len(test_eps), sorted(list(test_eps))[:3])
    _logger.info("Episode overlap: %d", len(dev_eps & test_eps))
    _logger.info("Dev speakers: %d, Test speakers: %d, Speaker overlap: %d", len(dev_spks), len(test_spks), len(dev_spks & test_spks))

    # Clustered block bootstrap by episode on real_test
    utterances = load_manifest(test_path)
    
    def load_hyps(run_id):
        p = Path("experiments") / run_id / "hyps_test.jsonl"
        hyps = []
        for line in open(p):
            if line.strip():
                hyps.append(json.loads(line)["hypothesis"])
        return hyps

    s0_hyps = load_hyps("S0-20261021-real-test-clean")
    s1_hyps = load_hyps("S1-20261021-real-test-clean")
    s4_hyps = load_hyps("S4-20261021-real-test-clean")

    s0_scores = score_pairs(utterances, s0_hyps)
    s1_scores = score_pairs(utterances, s1_hyps)
    s4_scores = score_pairs(utterances, s4_hyps)

    # Group scores by episode
    ep_clusters = defaultdict(list)
    for idx, u in enumerate(utterances):
        ep = u.utterance_id.rsplit("_", 1)[0]
        ep_clusters[ep].append(idx)

    ep_list = sorted(list(ep_clusters.keys()))
    _logger.info("Number of clusters (episodes) in test: %d", len(ep_list))

    rng = np.random.default_rng(42)
    B = 1000
    
    for metric_name, get_counts in [
        ("MER", lambda sc: sc.mer),
        ("SPER", lambda sc: sc.sper),
        ("SCR_en", lambda sc: ErrorCounts(sc.confusion.confusions_en, 0, 0, sc.confusion.eligible_en)),
    ]:
        for pair_name, sc_a, sc_b in [
            ("S4 vs S0", s4_scores, s0_scores),
            ("S1 vs S0", s1_scores, s0_scores),
            ("S4 vs S1", s4_scores, s1_scores),
        ]:
            diffs = []
            for _ in range(B):
                # sample clusters with replacement
                sample_eps = rng.choice(ep_list, size=len(ep_list), replace=True)
                sample_indices = [idx for ep in sample_eps for idx in ep_clusters[ep]]
                
                tot_a = ErrorCounts(0, 0, 0, 0)
                tot_b = ErrorCounts(0, 0, 0, 0)
                for idx in sample_indices:
                    ca = get_counts(sc_a[idx])
                    cb = get_counts(sc_b[idx])
                    tot_a = tot_a + ca
                    tot_b = tot_b + cb
                
                rate_a = (tot_a.errors / tot_a.reference_length) if tot_a.reference_length > 0 else 0.0
                rate_b = (tot_b.errors / tot_b.reference_length) if tot_b.reference_length > 0 else 0.0
                diffs.append((rate_a - rate_b) * 100.0)
            
            p_val_pos = np.mean([d >= 0 for d in diffs])
            p_val_neg = np.mean([d <= 0 for d in diffs])
            p_val = 2 * min(p_val_pos, p_val_neg)
            
            low = np.percentile(diffs, 2.5)
            high = np.percentile(diffs, 97.5)
            mean_diff = np.mean(diffs)
            _logger.info("Cluster Bootstrap (B=%d) | %s | %s | Diff: %+.2f%% | 95%% CI: [%+.2f%%, %+.2f%%] | p = %.4f",
                         B, pair_name, metric_name, mean_diff, low, high, p_val)

if __name__ == "__main__":
    main()
