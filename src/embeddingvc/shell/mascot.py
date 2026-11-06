"""The octopus mascot, drawn with half-block characters.

Terminals cannot play a GIF, so sprites.py holds its frames and the shell redraws them.
Each character cell shows two pixels, one above the other.
"""
from rich.style import Style
from rich.text import Text

from . import sprites


def draw(frame: int = 0) -> Text:
    """Fold pairs of pixel rows into half-block lines, as wide as the widest frame so nothing beside it shifts."""
    width = max(len(row) for rows in sprites.FRAMES for row in rows)
    pixels = [row.ljust(width, ".") for row in sprites.FRAMES[frame % len(sprites.FRAMES)]]
    text = Text(no_wrap=True)
    for top, bottom in zip(pixels[::2], pixels[1::2]):
        if text:
            text.append("\n")
        for upper, lower in zip(top, bottom):
            above, below = sprites.COLORS.get(upper), sprites.COLORS.get(lower)
            if above and below:
                text.append("▀", Style(color=above, bgcolor=below))
            elif above:
                text.append("▀", Style(color=above))
            elif below:
                text.append("▄", Style(color=below))
            else:
                text.append(" ")
    return text
