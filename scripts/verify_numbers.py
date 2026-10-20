#!/usr/bin/env python3
"""Inspect random sample of number-containing utterances to verify audio/script fidelity."""

import json
import re
import random

MANIFEST_PATH = "/home/oem/wiseyak_backup/wiseai-training-pipeline/datastore/nepali_english_codeswitched_podcasts/manifest.jsonl"
DIGIT_RE = re.compile(r"[\d\u0966-\u096F]")

records = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

number_recs = [r for r in records if DIGIT_RE.search(r.get("raw_devanagari_text", ""))]
print(f"Total utterances with numbers in raw: {len(number_recs)} / {len(records)}")

random.seed(42)
sample = random.sample(number_recs, min(25, len(number_recs)))

print("=" * 80)
print("SAMPLE 25 NUMBER UTTERANCES (RAW vs NORMALIZED GROUND TRUTH)")
print("=" * 80)

for idx, r in enumerate(sample, 1):
    utt = r["utterance_id"]
    dur = r["duration_seconds"]
    raw = r["raw_devanagari_text"]
    norm = r["text"]
    print(f"[{idx:02d}] {utt} ({dur:.2f}s)")
    print(f"     RAW : {raw}")
    print(f"     NORM: {norm}")
    print()
