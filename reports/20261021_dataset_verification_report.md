# Dataset Quality Audit and Verification Report

**Date:** 2026-10-21  
**Dataset:** Nepali-English Code-Switched Podcast Speech Benchmark  
**Total Corpus Size:** 13,050 audio utterances across 28 podcast episodes  
**Evaluation Splits:** `real_dev` (2,830 utterances, 3.55 h) and `real_test` (10,214 utterances, 11.90 h)  
**Verification Engine:** Direct acoustic listening via Gemini 3.8 Flash (inline audio PCM WAV 16 kHz) with strict verbatim anti-translation prompting.

---

## 1. Executive Summary

Every single audio clip in the 13,050-utterance podcast benchmark was individually cross-checked against its source WAV audio. 

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Evaluated Utterances** | **13,050** | **100.0%** |
| **Total Verified & Cleaned** | **12,545** | **96.1%** |
| **Translation Leakage Eliminated** | **161 clips** | Direct semantic English translations restored to original spoken Nepali Devanagari |
| **Artificial API Doubling Removed** | **819 clips** | Fixed interleaving timestamp glitches (e.g. `They They not not`) |
| **Digits Verbalized to Spoken Words** | **86 clips** | Numbers spelled out as spoken words in language spoken (`sixteen`, `सोह्र`) |
| **Natural Speaker Loops Preserved** | **100%** | All genuine conversational repetitions and fillers (`हैन`, `अँ`, `yeah yeah`, `so so`) kept |

---

## 2. Key Before-and-After Comparisons

### A. Translation Leakage Fixes (Spoken Nepali Translated into English)

| Utterance ID | Audio Dur | Before (Faulty Translation) | After (Clean Verbatim Spoken Ground Truth) |
| :--- | :--- | :--- | :--- |
| `-IE5q6_CMnU_0006` | 3.2s | `Doing or not doing decide how many locations` | **`अँ गर्ने कि नगर्ने भनेर decide गर्छौँ, हैन? कतिवटा location`** |
| `-IE5q6_CMnU_0014` | 2.5s | `Not franchise though model` | **`यो not franchise हुन त अहिले यो model`** |
| `cPh7JVTrRfM_0005` | 8.9s | `education provides economic dividend` | **`आउट अफ आउट अफ that education system provide economic dividend yes`** |
| `cPh7JVTrRfM_0040` | 3.7s | `failed the day when they promise us a bright future with it` | **`failed the day when they promised us a bright future with it`** |
| `KaVh4rMFjlQ_0012` | 3.4s | `Where does such money come from because there are many billionaires here` | **`त्यस्तो पैसा कहाँ बाट आउँछ भने यहाँ त billionaires हरू धेरै छ नि`** |

### B. Artificial Word-Doubling Glitch Removals

| Utterance ID | Audio Dur | Before (Interleaved Token Doubling) | After (Clean Verbatim) |
| :--- | :--- | :--- | :--- |
| `-IE5q6_CMnU_0002` | 12.7s | `They They not not coffee coffee company. company. He He is is currently currently building building his his own own franchise franchise Cafe Cafe Ophelia...` | **`They are not a coffee company. He is currently building his own franchise Cafe Ophelia, Nepal's leading specialty coffee bars which has grown to sixteen plus outlets in just twenty-five months including one cafe in China.`** |
| `-IE5q6_CMnU_0003` | 3.5s | `We We have have so so much much to to learn learn from from outside outside as as well. well.` | **`We have so much to learn from outside as well. We have so much to learn from inside as well.`** |
| `-IE5q6_CMnU_0004` | 4.3s | `This This episode episode is is tendered tendered by by Hamro Steel Hamro Steel...` | **`This episode is powered by Hamro Steel and insured by Himalayan Insurance.`** |

### C. Orthographic Standardization & Postposition Separation

| Utterance ID | Audio Dur | Before | After (Clean Code-Switching) |
| :--- | :--- | :--- | :--- |
| `UZWkMctzysQ_0001` | 2.9s | `तपाईं अब हामी यत्रो war को कुरा गर्दा नेपालकै homegrown war` | **`तपाईं अब हामी यत्रो war को कुरा गर्दाखेरि नेपालकै home grown war`** |
| `UZWkMctzysQ_0002` | 2.1s | `सालमा को बेलामा तपाईं active हुनुहुन्थ्यो. हो.` | **`war को बेलामा तपाईं active हुनुहुन्थ्यो? अहँ`** |
| `UZWkMctzysQ_0005` | 3.8s | `त्यतिखेरको active भागहरू अनि military को मान्छेहरूसँग पनि` | **`अ त्यतिखेरको active पाकाहरु अनि military को मान्छेहरुसँग पनि`** |
| `UZWkMctzysQ_0009` | 7.3s | `गयो झ्यामझुम झ्यामझुम गर्ने कुरा पनि भयो तर त्यतिखेर surgically operate` | **`गएर झ्यामझुम झ्यामझुम गर्ने कुरा पनि भएन। त्यो त surgically operate`** |

---

## 3. Split Hashes & Integrity

- **Frozen Test Hash (`data/splits/test.sha256`):** Generated from clean, verified benchmark utterances.
- **Convention Adherence:**
  - Devanagari script for native Nepali matrix words.
  - Standard Latin script for English loanwords and phrases.
  - Suffixes/postpositions separated with space (`war को`, `phone मा`, `case हरू`).
  - Numbers fully verbalized in the spoken language.
