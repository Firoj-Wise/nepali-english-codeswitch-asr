# Prompt: Nepali-English Code-Switched Sentence Generation

Use this prompt with the external text generator. Run it once per (domain, switch type) cell of the grid in section 2, with a fresh batch index each time. Save the outputs as JSON Lines in `data/text/sentences.raw.jsonl`, then run `scripts/validate_sentences.py`.

Keep two separate pools: one for synthesis and one for human recording (the reading set). Generate the reading set in its own runs, and check the two pools with `scripts/check_leakage.py` before using either.

## 1. Prompt text

Copy everything between the lines. Replace the bracketed fields.

---

You are generating training data for a speech recognition research project on Nepali-English code-switching, the way people in Nepal actually speak in daily life.

Write [N, for example 50] distinct sentences that a Nepali speaker might say aloud.

Domain: [DOMAIN]
Switch type: [SWITCH_TYPE]
Batch index: [BATCH] (use it to vary topics; do not repeat earlier batches)

Rules:

1. The sentence is mostly Nepali. English words appear where a real speaker would use them, such as technical words, brand-free product words, workplace terms, and common loanwords.
2. Write every Nepali word in Devanagari. Write every English word in Latin script, spelled correctly in English. Never transliterate an English word into Devanagari, and never romanize a Nepali word.
3. When a Nepali postposition or suffix follows an English word, write it as a separate word after a space: "phone मा", "meeting पछि", "office बाट".
4. Write numbers as words in the language in which they are spoken. Do not use digits.
5. Switch type rules:
   - single_word: exactly one English word in the sentence.
   - multi_word: two to four English words, not adjacent to each other.
   - phrase: one contiguous English phrase of two to five words.
6. Length: between 6 and 25 words.
7. Spoken register. Natural, colloquial, varied in person, tense, and politeness level (तपाईं, तिमी, हजुर).
8. No names of real people, companies, or brands. No phone numbers, addresses, or account numbers. No offensive content.
9. Every sentence must be different in structure, not only in a swapped word.
10. Output only JSON Lines, one object per line, with exactly these fields:
    {"sentence_id": "[DOMAIN]-[SWITCH_TYPE]-[BATCH]-<index>", "text": "<sentence>", "domain": "[DOMAIN]", "switch_type": "[SWITCH_TYPE]"}

---

## 2. Grid

Domains: banking, agriculture, health, education, telecom_support, insurance, government_services, shopping, travel, everyday_conversation.

Switch types: single_word, multi_word, phrase.

Suggested size for the synthesis pool: 10 domains by 3 switch types by about 700 sentences, roughly 20,000 sentences before validation. For the reading set: 10 domains by 3 switch types by about 40 sentences, roughly 1,200 sentences.

## 3. After generation

1. Run `scripts/validate_sentences.py` and read the rejection counts. A rejection rate above 20 percent means the prompt is being followed poorly. Fix the prompt before generating more.
2. Have a native speaker spot-check 100 random validated sentences for naturalness. Record the share judged natural in the data report.
3. Run `scripts/check_leakage.py` between the synthesis pool and the reading set.
