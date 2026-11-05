"""The owl that keeps the versions: pixel art drawn with half-block characters.

Terminals cannot play a GIF, so animation is a list of frames redrawn by Rich.
Each character cell shows two pixels, one above the other.
"""
from rich.style import Style
from rich.text import Text

COLORS = {"W": "#efeadf", "S": "#a7c7e7", "C": "#d8d2c4", "Y": "#f3dfa2", "E": "#2a2724", "O": "#d97757"}
EYES = {"open": ("YYYY", "YEEY", "YEEY", "YYYY"), "left": ("YYYY", "EEYY", "EEYY", "YYYY"),
        "right": ("YYYY", "YYEE", "YYEE", "YYYY"), "closed": ("WWWW", "WWWW", "EEEE", "WWWW")}
WINGS = {"down": (".SWWWWWWWWWWS.", ".SSWCCCCCCWSS.", ".SSWCCCCCCWSS."),
         "up": ("SSWWWWWWWWWWSS", "S.WCCCCCCCCW.S", "..WCCCCCCCCW..")}
BLANK = "." * 14
WORK_CYCLE = (("left", "down", 0), ("left", "up", 1), ("right", "down", 0), ("right", "up", 1),
              ("left", "down", 0), ("left", "up", 1), ("open", "down", 0), ("closed", "down", 0))


def sprite(eyes: str = "open", wings: str = "down") -> list[str]:
    ring = EYES[eyes]
    return ["..W........W..", "..WW......WW..", "..WWWWWWWWWW..",
            f".W{ring[0]}WW{ring[0]}W.", f".W{ring[1]}WW{ring[1]}W.",
            f".W{ring[2]}OO{ring[2]}W.", f".W{ring[3]}OO{ring[3]}W.",
            *WINGS[wings], "..SWWWWWWWWS..", "...OO....OO..."]


def draw(pixels: list[str]) -> Text:
    """Fold pairs of pixel rows into one line of half-block characters."""
    text = Text(no_wrap=True)
    for top, bottom in zip(pixels[::2], pixels[1::2]):
        if text:
            text.append("\n")
        for upper, lower in zip(top, bottom):
            above, below = COLORS.get(upper), COLORS.get(lower)
            if above and below:
                text.append("▀", Style(color=above, bgcolor=below))
            elif above:
                text.append("▀", Style(color=above))
            elif below:
                text.append("▄", Style(color=below))
            else:
                text.append(" ")
    return text


def still(eyes: str = "open") -> Text:
    return draw(sprite(eyes))


def working(frame: int) -> Text:
    """Flap, glance from side to side and blink; a one-pixel hop keeps it alive."""
    eyes, wings, hop = WORK_CYCLE[frame % len(WORK_CYCLE)]
    pixels = sprite(eyes, wings)
    return draw([*pixels, BLANK, BLANK] if hop else [BLANK, BLANK, *pixels])


def chunks(frame: int, width: int = 24) -> Text:
    """A stream of chunks drifting left, into the owl."""
    text = Text(no_wrap=True)
    for cell in range(width):
        slot = (cell + frame) % 12
        text.append({0: "▪", 4: "▪", 9: "■"}.get(slot, " "), ("#a8d8b9", "#c9b6e4", "#f3dfa2")[(cell + frame) // 12 % 3])
    return text
