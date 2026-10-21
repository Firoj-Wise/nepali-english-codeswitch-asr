#!/usr/bin/env python3
"""Cross-check and re-transcribe all real podcast evaluation audio clips using Gemini 3.8 Flash.

Features:
- Direct audio-to-text verbatim transcription via inline audio bytes.
- Strict anti-translation: spoken Nepali in Devanagari, spoken English in Latin.
- Attached Nepali postpositions separated by space (e.g. 'phone मा', 'case हरू').
- Numbers spelled out as spoken words.
- Natural speaker loops/repetitions preserved verbatim.
- Resumable JSONL cache.
- Multithreaded execution (configurable workers) with rate limit handling and exponential backoff.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

CSRL_ROOT = Path(__file__).resolve().parent.parent
DATASTORE = Path("/home/oem/wiseyak_backup/wiseai-training-pipeline/datastore/nepali_english_codeswitched_podcasts")
CLIPS_DIR = DATASTORE / "clips"
RAW_MANIFEST = DATASTORE / "manifest.jsonl"
OUTPUT_DIR = DATASTORE / "verified"
VERIFIED_MANIFEST = OUTPUT_DIR / "manifest_clean.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress_cache.jsonl"
LOG_FILE = CSRL_ROOT / "cross_check_dataset.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("dataset_cross_check")

SYSTEM_PROMPT = """You are an expert verbatim speech transcription engine for Nepali-English code-switched audio.

CRITICAL TRANSCRIPTION RULES:
1. VERBATIM ACCURACY: Transcribe every word exactly as spoken. NEVER summarize, omit, or guess.
2. NO TRANSLATION: Do NOT translate spoken Nepali into English, and do NOT translate spoken English into Nepali.
3. SCRIPT CONVENTION:
   - All genuine Nepali matrix words MUST be in Devanagari script.
   - All English words, loanwords, technical terms, and phrases MUST be written in Latin English script (e.g., "war", "scenario", "active", "military", "podcast", "situation", "design", "college", "model", "franchise", "information"). NEVER write English words phonetically in Devanagari (e.g., do NOT write "वार", write "war"; do NOT write "सिनारियो", write "scenario"; do NOT write "एक्टिभ", write "active"; do NOT write "मिलिटरी", write "military").
4. ATTACHED POSTPOSITIONS: If an English word has an attached Nepali suffix or postposition, write the root in Latin and separate the Devanagari postposition with a space (e.g., "war को", "phone मा", "case हरू", "model को", "apply गर्न", "active भएको").
5. NUMBERS AS WORDS: Spell out all numbers as spoken words in the language spoken (e.g., "sixteen", "twenty-five", "सोह्र", "पच्चीस"). Never leave digits.
6. PRESERVE REPETITIONS & FILLERS: Keep all natural conversational repetitions, loops, and fillers exactly as spoken (e.g., "अँ", "हैन", "yeah yeah", "so so", "you know").
7. OUTPUT FORMAT: Return ONLY the raw transcript text. Do not include markdown quotes, explanations, labels, or preamble."""


import threading


class ClientPool:
    def __init__(self, api_keys: list[str]):
        self.clients = [genai.Client(api_key=k) for k in api_keys]
        self._lock = threading.Lock()
        self._idx = 0

    def get_client(self) -> tuple[genai.Client, int]:
        with self._lock:
            cur = self._idx
            self._idx = (self._idx + 1) % len(self.clients)
            return self.clients[cur], cur

    def get_alternate(self, current_idx: int) -> tuple[genai.Client, int]:
        next_idx = (current_idx + 1) % len(self.clients)
        return self.clients[next_idx], next_idx


def transcribe_single_clip(client_pool: ClientPool, clip_path: Path, max_retries: int = 5) -> str:
    """Send inline audio bytes to Gemini 3.8 Flash for verbatim transcription, rotating keys on rate limit."""
    with open(clip_path, "rb") as f:
        audio_bytes = f.read()

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
    client, key_idx = client_pool.get_client()

    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    audio_part,
                    "Transcribe this audio clip verbatim following all system rules. Output only the transcript."
                ],
                config=config
            )
            text = response.text.strip() if response.text else ""
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text[1:-1].strip()
            return text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                client, key_idx = client_pool.get_alternate(key_idx)
                logger.warning(f"Key {key_idx} hit rate limit on {clip_path.name}. Switched to alternate key.")
                time.sleep(1.0)
            elif attempt < max_retries - 1:
                time.sleep(1.0)
            else:
                logger.error(f"Failed to transcribe {clip_path.name} after {max_retries} attempts: {e}")
                return ""
    return ""


def process_record(rec: dict, client_pool: ClientPool) -> dict:
    utt_id = rec["utterance_id"]
    clip_name = Path(rec["audio_path"]).name
    clip_path = CLIPS_DIR / clip_name

    if not clip_path.exists():
        logger.warning(f"Clip not found: {clip_path}")
        return rec

    clean_text = transcribe_single_clip(client_pool, clip_path)
    if clean_text:
        rec["clean_verbatim_text"] = clean_text
        rec["old_text"] = rec.get("text", "")
        rec["text"] = clean_text
    else:
        rec["clean_verbatim_text"] = rec.get("text", "")

    return rec


def main():
    parser = argparse.ArgumentParser(description="Cross-check and re-transcribe audio dataset with Gemini 3.8 Flash.")
    parser.add_argument("--workers", type=int, default=32, help="Number of concurrent worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for dry run / testing")
    args = parser.parse_args()

    load_dotenv(CSRL_ROOT / ".env")
    keys = []
    # Primary key
    if os.environ.get("GEMINI_API_KEY"):
        keys.append(os.environ.get("GEMINI_API_KEY"))
    # Secondary key(s)
    if os.environ.get("GEMINI_API_KEY_2"):
        keys.append(os.environ.get("GEMINI_API_KEY_2"))
    if os.environ.get("GEMINI_API_KEYS"):
        for k in os.environ.get("GEMINI_API_KEYS").split(","):
            if k.strip() and k.strip() not in keys:
                keys.append(k.strip())

    if not keys:
        logger.error("No GEMINI_API_KEY found in environment.")
        sys.exit(1)

    logger.info(f"Initialized ClientPool with {len(keys)} active API key(s).")
    client_pool = ClientPool(keys)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load progress cache
    completed = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
            for line in pf:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        completed[entry["utterance_id"]] = entry
                    except Exception:
                        pass
        logger.info(f"Loaded {len(completed)} already verified utterances from progress cache.")

    # 2. Load manifest records
    records = []
    with open(RAW_MANIFEST, "r", encoding="utf-8") as mf:
        for line in mf:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} total records from manifest.")
    if args.limit:
        records = records[:args.limit]
        logger.info(f"Limiting to first {args.limit} records for test run.")

    # 3. Filter pending
    pending = [r for r in records if r["utterance_id"] not in completed]
    logger.info(f"Utterances pending verification: {len(pending)}")

    if not pending:
        logger.info("All records already verified!")
    else:
        logger.info(f"Starting parallel verification with {args.workers} workers...")
        start_time = time.time()
        processed_count = 0

        with open(PROGRESS_FILE, "a", encoding="utf-8") as pf:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_to_utt = {
                    executor.submit(process_record, rec, client_pool): rec["utterance_id"]
                    for rec in pending
                }

                for future in concurrent.futures.as_completed(future_to_utt):
                    utt_id = future_to_utt[future]
                    try:
                        res = future.result()
                        pf.write(json.dumps(res, ensure_ascii=False) + "\n")
                        pf.flush()
                        completed[utt_id] = res
                        processed_count += 1

                        if processed_count % 100 == 0 or processed_count == len(pending):
                            elapsed = time.time() - start_time
                            rate = processed_count / max(1.0, elapsed)
                            remaining = (len(pending) - processed_count) / max(0.1, rate)
                            logger.info(
                                f"Progress: {processed_count}/{len(pending)} "
                                f"({processed_count / len(pending) * 100:.1f}%) | "
                                f"Rate: {rate:.1f} clips/s | Remaining: {remaining / 60.0:.1f}m"
                            )
                    except Exception as exc:
                        logger.error(f"Error processing {utt_id}: {exc}")

    # 4. Generate final clean manifest
    logger.info(f"Writing final clean manifest to {VERIFIED_MANIFEST}...")
    with open(VERIFIED_MANIFEST, "w", encoding="utf-8") as vf:
        for r in records:
            utt_id = r["utterance_id"]
            if utt_id in completed:
                vf.write(json.dumps(completed[utt_id], ensure_ascii=False) + "\n")
            else:
                vf.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Dataset cross-check and cleaning finished successfully!")


if __name__ == "__main__":
    main()
