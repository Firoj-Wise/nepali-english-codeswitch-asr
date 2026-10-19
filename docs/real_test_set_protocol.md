# Real Test Set Protocol

The real recordings are the only evaluation data for every claim in the paper. They are never used for training, filtering thresholds, or prompt design.

## 1. Target

- 1 to 2 hours of speech in total.
- At least 20 speakers, balanced by gender as far as possible, with a range of ages and regions.
- Two parts:
  - **Read:** speakers read sentences from the reading set.
  - **Spontaneous:** speakers answer short prompts, for example "describe how you pay your bills". These are transcribed by hand.

## 2. Consent

Every speaker signs a consent form before recording. The form states:

- what the recordings are for: research on speech recognition, a publication, and a possible public release of the test set;
- who can access the recordings;
- that the speaker may withdraw.

Store signed forms outside this repository. Record only an anonymized speaker id in manifests. Without consent for release, the test set stays internal, and the paper says so.

## 3. Recording

- Phone or laptop microphones in ordinary rooms. Real conditions are the point.
- Record at 16 kHz or higher. Convert to 16 kHz mono WAV before use.
- Note the device type per speaker in a side file. Do not put device data in the manifest text.

## 4. Transcription

- Follow `docs/data_spec.md` section 4 exactly.
- Read speech: start from the prompt sentence and correct it to what the speaker actually said, including disfluencies that are words.
- Spontaneous speech: two independent transcribers per utterance on at least 20 percent of the data. Report the inter-annotator agreement (MER between the two transcriptions) in the data report.
- Utterances that cannot be transcribed with confidence are excluded and counted.

## 5. Split and freeze

`scripts/prepare_splits.py` splits the recordings by speaker into real dev (model selection) and real test (final numbers), and freezes the real test hash. Real dev and real test never share a speaker.
