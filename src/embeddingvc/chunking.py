"""Deterministic recursive-character chunking.

The same text and the same settings always produce the same chunks, in the
same order, with the same offsets. Staging depends on it: an unchanged file
must re-hash to the objects already stored, or every rescan would look like a
modification.

Separators are tried in order from the largest natural boundary to the
smallest, so a paragraph is only broken at a newline when the paragraph itself
exceeds the chunk size, and only broken mid-word as a last resort.

Chunks are produced in two passes. Splitting reduces the text to pieces that
each fit, remembering where each began; merging then walks those pieces
forward, emitting a chunk whenever the next piece would overflow. A merged
chunk is sliced straight out of the source rather than rebuilt by joining
pieces, which keeps its offsets exact and its separators identical to the
original.
"""

from dataclasses import dataclass

#: Paragraph, line, word, then character. The empty separator is the fallback
#: that guarantees termination for text with no whitespace at all.
DEFAULT_SEPARATORS = ("\n\n", "\n", " ", "")


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int


def _split(text: str, base: int, separators: tuple[str, ...], limit: int) -> list[tuple[str, int]]:
    """Reduce text to (piece, offset) pairs, each within `limit` where possible."""
    if not text:
        return []
    if len(text) <= limit:
        return [(text, base)]
    if not separators:
        return [(text, base)]

    separator, remaining = separators[0], separators[1:]
    if separator == "":
        return [(text[index:index + limit], base + index) for index in range(0, len(text), limit)]
    if separator not in text:
        return _split(text, base, remaining, limit)

    pieces: list[tuple[str, int]] = []
    cursor = 0
    for part in text.split(separator):
        if part:
            if len(part) <= limit:
                pieces.append((part, base + cursor))
            else:
                pieces.extend(_split(part, base + cursor, remaining, limit))
        cursor += len(part) + len(separator)
    return pieces


def _tail_within(window: list[tuple[str, int]], overlap: int) -> list[tuple[str, int]]:
    """Return the trailing pieces that fit inside `overlap` characters."""
    if overlap <= 0 or not window:
        return []
    end = window[-1][1] + len(window[-1][0])
    tail: list[tuple[str, int]] = []
    for piece in reversed(window):
        if end - piece[1] > overlap:
            break
        tail.insert(0, piece)
    return tail


def split_text(text: str, chunk_size: int, chunk_overlap: int,
               separators: tuple[str, ...] = DEFAULT_SEPARATORS) -> list[Chunk]:
    """Split text into overlapping chunks, in reading order.

    Empty or whitespace-only text yields no chunks, which is how an empty file
    becomes a tracked document with zero occurrences rather than an error.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must not be negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
    if not text.strip():
        return []

    pieces = _split(text, 0, tuple(separators), chunk_size)
    if not pieces:
        return []

    chunks: list[Chunk] = []
    window: list[tuple[str, int]] = []

    def emit(current: list[tuple[str, int]]) -> None:
        start = current[0][1]
        end = current[-1][1] + len(current[-1][0])
        chunks.append(Chunk(text=text[start:end], start=start, end=end))

    for piece in pieces:
        if window:
            reach = piece[1] + len(piece[0])
            if reach - window[0][1] > chunk_size:
                emit(window)
                window = _tail_within(window, chunk_overlap)
                # Trim any carried-over context that would push this piece past
                # the limit; without this a large piece could never start a
                # chunk and the loop would emit the same window forever.
                while window and reach - window[0][1] > chunk_size:
                    window.pop(0)
        window.append(piece)

    if window:
        emit(window)
    return chunks
