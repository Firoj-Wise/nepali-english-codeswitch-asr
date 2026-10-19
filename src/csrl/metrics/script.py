"""Script labelling of tokens."""

from csrl.domain import constants
from csrl.domain.enums import Script


def _char_script(character: str) -> Script:
    code_point = ord(character)
    if constants.DEVANAGARI_BLOCK_START <= code_point <= constants.DEVANAGARI_BLOCK_END:
        return Script.DEVANAGARI
    if character.isascii() and character.isalpha():
        return Script.LATIN
    return Script.OTHER


def token_script(token: str) -> Script:
    """Label a token per docs/metrics_spec.md section 2."""
    letter_scripts = {
        _char_script(c) for c in token if c.isalpha() or _char_script(c) is Script.DEVANAGARI
    }
    letter_scripts.discard(Script.OTHER)
    has_ascii_digit = any(c in constants.ASCII_DIGITS for c in token)
    if not letter_scripts:
        return Script.NUMERIC if has_ascii_digit else Script.OTHER
    if len(letter_scripts) > 1:
        return Script.MIXED
    return next(iter(letter_scripts))
