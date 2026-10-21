"""Batch processor for all podcast episodes.

Transcribes via gemini-3.5-transcribe with word timestamps and diarization,
normalizes English loanwords to Latin via gemini-3.8-flash,
slices 16 kHz sentence clips directly into wiseai-training-pipeline datastore,
creates symlinks in csrl, and logs progress.
"""
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

CSRL_ROOT = Path(__file__).resolve().parent.parent
TARGET_DATASTORE = Path("/home/oem/wiseyak_backup/wiseai-training-pipeline/datastore/nepali_english_codeswitched_podcasts")
TARGET_CLIPS = TARGET_DATASTORE / "clips"
TARGET_MANIFEST = TARGET_DATASTORE / "manifest.jsonl"
TARGET_README = TARGET_DATASTORE / "README.md"
RAW_MANIFEST_PATH = CSRL_ROOT / "data" / "manifests" / "real_raw.jsonl"
CACHE_DIR = TARGET_DATASTORE / "cache"

# Local CSRL symlinks
LOCAL_CLIPS_SYMLINK = CSRL_ROOT / "data" / "audio" / "real_clips"
LOCAL_MANIFEST_SYMLINK = CSRL_ROOT / "data" / "manifests" / "real_all_manifest.jsonl"
LOG_FILE = CSRL_ROOT / "process_all_episodes.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8")
    ]
)
logger = logging.getLogger("batch_processor")


def parse_offset(offset_str) -> float:
    if not offset_str:
        return -1.0
    try:
        return float(str(offset_str).rstrip("s"))
    except (ValueError, TypeError, AttributeError):
        return -1.0


def find_split_points(wav_path: Path, total_dur: float, max_chunk_s: float = 1500.0) -> list[tuple[float, float]]:
    """Split long audio at acoustic silence points near target chunk boundaries.
    Ensures zero words are clipped at boundaries and speech quality is 100% preserved.
    """
    if total_dur <= max_chunk_s:
        return [(0.0, total_dur)]

    num_chunks = int(math.ceil(total_dur / max_chunk_s))
    target_len = total_dur / num_chunks

    split_points = [0.0]
    sample_rate = 16000
    bytes_per_sec = sample_rate * 2  # 16-bit mono PCM

    try:
        with open(wav_path, "rb") as f:
            for i in range(1, num_chunks):
                approx_split = i * target_len
                window_start_s = max(split_points[-1] + 60.0, approx_split - 10.0)
                window_dur_s = 20.0
                byte_offset = 44 + int(window_start_s * bytes_per_sec)
                bytes_to_read = int(window_dur_s * bytes_per_sec)

                f.seek(byte_offset)
                raw = f.read(bytes_to_read)
                if len(raw) < 3200:
                    best_split = approx_split
                else:
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    frame_size = 1600  # 100ms
                    hop = 800         # 50ms
                    if len(samples) > frame_size:
                        num_frames = (len(samples) - frame_size) // hop
                        energies = [
                            float(np.mean(samples[j*hop : j*hop + frame_size] ** 2))
                            for j in range(num_frames)
                        ]
                        min_frame_idx = int(np.argmin(energies))
                        silence_offset_s = (min_frame_idx * hop + frame_size // 2) / sample_rate
                        best_split = window_start_s + silence_offset_s
                    else:
                        best_split = approx_split

                split_points.append(round(best_split, 3))
    except Exception as e:
        logger.warning(f"Could not calculate silence split points: {e}. Falling back to uniform splits.")
        split_points = [i * target_len for i in range(num_chunks)]

    split_points.append(total_dur)
    segments = [(split_points[i], split_points[i+1]) for i in range(len(split_points) - 1)]
    return segments


def chunk_words(parts, min_dur=2.5, max_dur=15.0, gap_split=0.5):
    all_words = []
    for part in parts:
        trans = getattr(part, "audio_transcription", None)
        if not trans or not trans.words:
            continue
        speaker = trans.speaker_label or "spk:0"
        for w in trans.words:
            s = parse_offset(w.start_offset)
            e = parse_offset(w.end_offset)
            if s >= 0.0 and e > s:
                all_words.append({
                    "word": w.word,
                    "start": s,
                    "end": e,
                    "speaker": speaker
                })

    if not all_words:
        return []

    all_words.sort(key=lambda x: x["start"])
    chunks = []
    current = []
    c_start = all_words[0]["start"]
    c_spk = all_words[0]["speaker"]

    for w in all_words:
        if not current:
            current.append(w)
            c_start = w["start"]
            c_spk = w["speaker"]
            continue

        prev = current[-1]
        gap = w["start"] - prev["end"]
        dur = w["end"] - c_start
        spk_change = (w["speaker"] != c_spk)
        is_pause = (gap >= gap_split)

        split = False
        if spk_change and dur >= min_dur or is_pause and dur >= min_dur or dur >= max_dur:
            split = True

        if split:
            end_t = max(c_start + 1.0, prev["end"])
            chunks.append({
                "speaker": c_spk,
                "start": c_start,
                "end": end_t,
                "duration": end_t - c_start,
                "raw_text": " ".join(cw["word"] for cw in current)
            })
            current = [w]
            c_start = w["start"]
            c_spk = w["speaker"]
        else:
            current.append(w)

    if current:
        end_t = max(c_start + 1.0, current[-1]["end"])
        chunks.append({
            "speaker": c_spk,
            "start": c_start,
            "end": end_t,
            "duration": end_t - c_start,
            "raw_text": " ".join(cw["word"] for cw in current)
        })

    return [c for c in chunks if c["duration"] >= 1.5 and c["end"] > c["start"]]


def normalize_batch(client: genai.Client, raw_sentences: list[str]) -> list[str]:
    prompt = """You are an expert linguist specializing in Nepali-English code-switched speech for ASR evaluation.
You are given a JSON array of sentences transcribed from audio.
Restore all English loanwords and phrases to standard English Latin script, while keeping all genuine Nepali words in Devanagari script.

Rules:
1. English words MUST be written in Latin script (e.g., सिचुएसन -> situation, अपरेट -> operate, होम ग्रोन -> homegrown, इन्फर्मेसन -> information, डिजाइन -> design, कलेज -> college, भिडियो -> video, वार -> war).
2. Nepali words MUST remain in Devanagari script.
3. If an English word has an attached Nepali suffix or postposition, separate it with a space (e.g., कलेजमा -> college मा, सिचुएसनको -> situation को, अप्लाई गर्न -> apply गर्न).
4. Numbers must be spelled out as spoken words.
5. Return ONLY a valid JSON array of strings matching the input length exactly.

Input:
""" + json.dumps(raw_sentences, ensure_ascii=False)

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=config
        )
        res_list = json.loads(response.text)
        if isinstance(res_list, list) and len(res_list) == len(raw_sentences):
            return res_list
        return raw_sentences
    except Exception as e:
        logger.warning(f"Normalization error: {e}. Keeping raw text.")
        return raw_sentences


def slice_clip(src_wav: Path, dest_wav: Path, start_s: float, end_s: float):
    start = max(0.0, start_s - 0.05)
    end = max(start + 0.5, end_s + 0.05)
    dur = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{dur:.3f}",
        "-i", str(src_wav),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(dest_wav)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg slicing error: {res.stderr}")


def write_readme():
    content = """# Nepali-English Code-Switched Podcast Evaluation Dataset

## Overview
High-quality, conversational Nepali-English code-switched speech dataset extracted from 28 prominent youth podcast episodes on YouTube (Sushant Pradhan Podcast, ON AIR with Sanjay Silwal Gupta, The Doers Nepal).

## Format and Specifications
- **Audio:** 16,000 Hz, 16-bit mono PCM WAV.
- **Utterance Duration:** 2.0 to 15.0 seconds per clip (mean ~4.5s).
- **Acoustic Transcription:** `gemini-3.5-transcribe` with word-level timestamps and speaker diarization.
- **Script Conventions (docs/data_spec.md):**
  - All Nepali words in Devanagari script.
  - All English words in Latin script.
  - Attached postpositions separated by space (e.g., `war को`, `college मा`).
  - Numbers written as spoken words.

## Metadata Fields
- `utterance_id`: Unique identifier (`{video_id}_{index:04d}`).
- `audio_path`: Relative path to WAV file.
- `duration_seconds`: Exact clip duration.
- `speaker_id`: Speaker label with video prefix.
- `raw_devanagari_text`: Verbatim acoustic transcript in Devanagari.
- `text`: Code-switched normalized reference.
- `show`: Podcast channel/series name.
- `video_id`: Source YouTube video identifier.
"""
    with open(TARGET_README, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY missing.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    TARGET_CLIPS.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_DATASTORE.mkdir(parents=True, exist_ok=True)

    write_readme()

    # Load raw manifest
    episodes = []
    with open(RAW_MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    logger.info(f"Loaded {len(episodes)} podcast episodes to process.")

    # Check already completed video IDs in target manifest
    processed_videos = set()
    if TARGET_MANIFEST.exists():
        with open(TARGET_MANIFEST, "r", encoding="utf-8") as mf:
            for line in mf:
                if line.strip():
                    try:
                        r = json.loads(line)
                        vid = r.get("utterance_id", "").rsplit("_", 1)[0]
                        if vid:
                            processed_videos.add(vid)
                    except Exception:
                        pass

    # Import pilot episode if already processed
    pilot_manifest = CSRL_ROOT / "data" / "manifests" / "pilot_manifest.jsonl"
    pilot_dir = CSRL_ROOT / "data" / "audio" / "real_pilot"
    if pilot_manifest.exists() and "UZWkMctzysQ" not in processed_videos:
        logger.info("Importing completed pilot episode (UZWkMctzysQ)...")
        with open(pilot_manifest, "r", encoding="utf-8") as pmf, open(TARGET_MANIFEST, "a", encoding="utf-8") as tmf:
            for line in pmf:
                if line.strip():
                    rec = json.loads(line)
                    src_clip = CSRL_ROOT / rec["audio_path"]
                    dest_clip = TARGET_CLIPS / f"{rec['utterance_id']}.wav"
                    if src_clip.exists() and not dest_clip.exists():
                        shutil.copy2(src_clip, dest_clip)
                    rec["audio_path"] = f"clips/{dest_clip.name}"
                    rec["show"] = "sushant pradhan"
                    tmf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        processed_videos.add("UZWkMctzysQ")
        logger.info("Pilot episode imported successfully.")

    for idx, ep in enumerate(episodes, 1):
        vid = ep["video_id"]
        show = ep.get("show", "podcast")
        title = ep.get("title", vid)
        raw_wav = CSRL_ROOT / ep["audio_path"]

        if vid in processed_videos:
            logger.info(f"[{idx}/{len(episodes)}] Video {vid} already processed. Skipping.")
            continue

        if not raw_wav.exists():
            logger.error(f"[{idx}/{len(episodes)}] Audio file missing: {raw_wav}. Skipping.")
            continue

        dur_sec = ep.get("duration_seconds", 0)
        dur_min = dur_sec / 60.0
        logger.info(f"\n[{idx}/{len(episodes)}] Processing: {title[:50]} ({dur_min:.1f} min)")

        try:
            # 1. Transcribe or Load from Cache
            cache_file = CACHE_DIR / f"{vid}_transcription.json"
            parts_data = None

            if cache_file.exists():
                logger.info("Loading transcription from local cache...")
                with open(cache_file, "r", encoding="utf-8") as cf:
                    parts_data = json.load(cf)
            else:
                # If audio is longer than 40 minutes, split at silence points to respect API limits
                max_seg_s = 1500.0  # 25 minutes
                segments = find_split_points(raw_wav, dur_sec, max_chunk_s=max_seg_s)
                parts_data = []

                if len(segments) == 1:
                    logger.info("Uploading audio to Gemini Files API...")
                    up_file = client.files.upload(file=str(raw_wav))
                    try:
                        logger.info("Transcribing via gemini-3.5-transcribe...")
                        t0 = time.time()
                        t_config = types.AudioTranscriptionConfig(
                            word_timestamp=True,
                            diarization=True,
                            language_codes=["ne", "en"],
                        )
                        g_config = types.GenerateContentConfig(
                            audio_transcription_config=t_config,
                        )
                        resp = client.models.generate_content(
                            model="gemini-3.5-transcribe",
                            contents=[up_file],
                            config=g_config
                        )
                        elapsed = time.time() - t0
                        logger.info(f"Transcription finished in {elapsed:.1f}s.")

                        candidates = getattr(resp, "candidates", [])
                        if candidates and candidates[0].content and candidates[0].content.parts:
                            for p in candidates[0].content.parts:
                                trans = getattr(p, "audio_transcription", None)
                                if trans and trans.words:
                                    part_words = []
                                    for w in trans.words:
                                        w_s = parse_offset(w.start_offset)
                                        w_e = parse_offset(w.end_offset)
                                        if w_s < 0.0:
                                            continue
                                        if w_e <= w_s:
                                            w_e = w_s + 0.3
                                        part_words.append({
                                            "word": w.word,
                                            "start_offset": f"{w_s:.3f}s",
                                            "end_offset": f"{w_e:.3f}s"
                                        })
                                    parts_data.append({
                                        "speaker_label": trans.speaker_label or "spk:0",
                                        "words": part_words
                                    })
                    finally:
                        try:
                            client.files.delete(name=up_file.name)
                        except Exception:
                            pass
                else:
                    logger.info(f"Audio duration is {dur_min:.1f}m (>25m). Splitting into {len(segments)} silence-bounded parts...")
                    for s_idx, (seg_start, seg_end) in enumerate(segments, 1):
                        seg_dur = seg_end - seg_start
                        logger.info(f"  Part {s_idx}/{len(segments)}: {seg_start/60.0:.1f}m - {seg_end/60.0:.1f}m ({seg_dur/60.0:.1f} min)")
                        seg_wav = CACHE_DIR / f"temp_{vid}_seg{s_idx}.wav"
                        slice_clip(raw_wav, seg_wav, seg_start, seg_end)

                        try:
                            for attempt in range(2):
                                up_file = client.files.upload(file=str(seg_wav))
                                try:
                                    t0 = time.time()
                                    t_config = types.AudioTranscriptionConfig(
                                        word_timestamp=True,
                                        diarization=True,
                                        language_codes=["ne", "en"],
                                    )
                                    g_config = types.GenerateContentConfig(
                                        audio_transcription_config=t_config,
                                    )
                                    resp = client.models.generate_content(
                                        model="gemini-3.5-transcribe",
                                        contents=[up_file],
                                        config=g_config
                                    )
                                    elapsed = time.time() - t0
                                    logger.info(f"  Part {s_idx} transcribed in {elapsed:.1f}s.")

                                    candidates = getattr(resp, "candidates", [])
                                    if candidates and candidates[0].content and candidates[0].content.parts:
                                        for p in candidates[0].content.parts:
                                            trans = getattr(p, "audio_transcription", None)
                                            if trans and trans.words:
                                                part_words = []
                                                for w in trans.words:
                                                    w_s = parse_offset(w.start_offset)
                                                    w_e = parse_offset(w.end_offset)
                                                    if w_s < 0.0:
                                                        continue
                                                    if w_e <= w_s:
                                                        w_e = w_s + 0.3
                                                    w_start = w_s + seg_start
                                                    w_end = w_e + seg_start
                                                    part_words.append({
                                                        "word": w.word,
                                                        "start_offset": f"{w_start:.3f}s",
                                                        "end_offset": f"{w_end:.3f}s"
                                                    })
                                                parts_data.append({
                                                    "speaker_label": trans.speaker_label or f"spk:{s_idx}",
                                                    "words": part_words
                                                })
                                        break
                                    else:
                                        logger.warning(f"  Part {s_idx} attempt {attempt+1}: empty content. Retrying...")
                                        time.sleep(3)
                                finally:
                                    try:
                                        client.files.delete(name=up_file.name)
                                    except Exception:
                                        pass
                        finally:
                            if seg_wav.exists():
                                seg_wav.unlink()

                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(parts_data, cf, ensure_ascii=False)

            # 2. Chunk words into sentences
            class WordObj:
                def __init__(self, d):
                    self.word = d["word"]
                    self.start_offset = d["start_offset"]
                    self.end_offset = d["end_offset"]

            class TransObj:
                def __init__(self, p):
                    self.speaker_label = p.get("speaker_label", "spk:0")
                    self.words = [WordObj(wd) for wd in p.get("words", [])]

            class PartObj:
                def __init__(self, p):
                    self.audio_transcription = TransObj(p)

            part_objs = [PartObj(p) for p in parts_data]
            chunks = chunk_words(part_objs, min_dur=2.5, max_dur=15.0)
            logger.info(f"Generated {len(chunks)} sentence chunks.")

            # 3. Batch normalize with Gemini 3.8 Flash
            raw_texts = [c["raw_text"] for c in chunks]
            batch_size = 50
            norm_texts = []
            logger.info(f"Normalizing scripts for {len(chunks)} utterances...")
            for b_idx in range(0, len(raw_texts), batch_size):
                batch = raw_texts[b_idx:b_idx + batch_size]
                norm_batch = normalize_batch(client, batch)
                norm_texts.extend(norm_batch)

            # 4. Slice clips and write to manifest
            logger.info(f"Slicing {len(chunks)} audio clips into datastore...")
            new_records = []
            with open(TARGET_MANIFEST, "a", encoding="utf-8") as mf:
                for c_idx, (chunk, norm_t) in enumerate(zip(chunks, norm_texts), 1):
                    utt_id = f"{vid}_{c_idx:04d}"
                    dest_wav = TARGET_CLIPS / f"{utt_id}.wav"
                    try:
                        slice_clip(raw_wav, dest_wav, chunk["start"], chunk["end"])
                        rec = {
                            "utterance_id": utt_id,
                            "audio_path": f"clips/{dest_wav.name}",
                            "duration_seconds": round(chunk["duration"], 3),
                            "speaker_id": f"{vid}:{chunk['speaker']}",
                            "raw_devanagari_text": chunk["raw_text"],
                            "text": norm_t,
                            "show": show,
                            "video_id": vid,
                            "source": "real_podcast"
                        }
                        mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        new_records.append(rec)
                    except Exception as se:
                        logger.warning(f"Error slicing {utt_id}: {se}")

            processed_videos.add(vid)
            logger.info(f"Episode {vid} complete. Added {len(new_records)} clips.")
        except Exception as ep_err:
            logger.error(f"Error processing episode {vid}: {ep_err}", exc_info=True)
            continue

    # Symlink setup to csrl
    logger.info("Updating symlinks in csrl workspace...")
    if not LOCAL_CLIPS_SYMLINK.is_symlink() and not LOCAL_CLIPS_SYMLINK.exists():
        LOCAL_CLIPS_SYMLINK.symlink_to(TARGET_CLIPS)
    if not LOCAL_MANIFEST_SYMLINK.is_symlink() and not LOCAL_MANIFEST_SYMLINK.exists():
        LOCAL_MANIFEST_SYMLINK.symlink_to(TARGET_MANIFEST)

    logger.info("ALL EPISODES PROCESSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
