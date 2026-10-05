"""Pure text and candidate checks shared by collection and runtime.

No environment changes, dataset preparation or model loading on import.
Normalization is for source isolation / length checks, never no-op decisions.
"""
import hashlib
import re
import unicodedata


def norm(text):
    return ''.join(unicodedata.normalize('NFKC', text).casefold().split())


def source_key(text):
    return hashlib.sha256(norm(text).encode()).hexdigest()


def valid_candidate(before, generation):
    text = generation['text'].strip()
    if generation['finish_reason'] == 'context_length_exceeded':
        return False, 'context_length_exceeded'
    if not text:
        return False, 'empty'
    if generation['finish_reason'] != 'stop':
        return False, 'unfinished'
    if any(x in text.lower() for x in ('<think>', '</think>', '```')):
        return False, 'format'
    if not re.search(r'[\u3400-\u9fff]', text):
        return False, 'no_chinese'
    # Basic safety gate only; never select by feedback or by estimated quality.
    if before and len(norm(text)) > 1.5 * len(norm(before)) + 8:
        return False, 'length'
    return True, 'valid'

