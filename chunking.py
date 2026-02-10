import re
import hashlib


def preprocess_text(text: str) -> str:
    """Clean text: remove HTML, normalize whitespace, quotes, non-printable characters"""
    text = re.sub(r"</?[^>]+(>|$)", "", text)  # strip HTML
    text = re.sub(r"[\r\n]+", " ", text)  # normalize newlines
    text = re.sub(r"\s+", " ", text)  # collapse spaces/tabs
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^\x20-\x7E]+", "", text)  # remove non-ASCII
    return text.strip()


def chunk_text(text: str, sentences_per_chunk=2, overlap=1):
    """Split text into chunks of sentences with optional overlap"""
    text = preprocess_text(text)
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks = []

    i = 0
    chunk_id = 0
    while i < len(sentences):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        if not chunk_sentences:
            break
        prev_context = sentences[i - 1] if i > 0 else ""
        chunks.append(
            {
                "text": " ".join(chunk_sentences),
                "metadata": {
                    "chunk_id": chunk_id,
                    "start_sentence": i,
                    "end_sentence": i + len(chunk_sentences) - 1,
                    "previous_context": prev_context,
                },
            }
        )
        chunk_id += 1
        i += max(1, sentences_per_chunk - overlap)
    return chunks


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
