# Writing Style

Applies to README files, docs, reports, docstrings, comments, and commit messages.

## Mechanical rules (checked by `scripts/check_repo.py`)

1. No em dash (U+2014) or en dash (U+2013). Use a comma, colon, parentheses, or a new sentence. Number ranges are written "10 to 20".
2. No emoji or pictographic symbols.
3. No exclamation marks in prose.
4. No banned phrases from `scripts/banned_phrases.txt`.

## Content rules

1. Write as a researcher writes a methods section: what was done, with which data, measured how, with what result.
2. Every quantitative claim carries its run id, split, metric version, and confidence interval.
3. Distinguish observation from interpretation. Use "we observe" for data and "this suggests" for interpretation.
4. No intensifiers without evidence ("significantly" only with a test that supports it).
5. Short sentences. One idea per paragraph.
6. Limitations are stated in their own section in every report.
