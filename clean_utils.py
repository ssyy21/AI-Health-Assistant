# clean_utils.py
import re

def normalize_text(text: str) -> str:
    """Lowercase, remove extra spaces + unicode normalization."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text
