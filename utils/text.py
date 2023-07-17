import numpy as np
from g2pk import G2p

HANGUL_JAMO_INITIAL = list(chr(i) for i in range(0x1100, 0x1113))
HANGUL_JAMO_MEDIAL = list(chr(i) for i in range(0x1161, 0x1176))
HANGUL_JAMO_FINAL = list(chr(i) for i in range(0x11A8, 0x11C3))
SPECIAL_CHARACTERS = ["<PAD>"]
CHARACTERS = (
    HANGUL_JAMO_INITIAL + HANGUL_JAMO_MEDIAL + HANGUL_JAMO_FINAL + SPECIAL_CHARACTERS
)
CHAR_TO_CODE = {char: i for i, char in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS)
g2p = G2p()


def split_hangul_jamo(text: str) -> list[str]:
    """Splits Hangul into individual characters.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Split text."""
    jamo_list = []
    for char in text:
        if 0xAC00 <= ord(char) <= 0xD7A3:
            jamo_list.append(chr(0x1100 + (ord(char) - 0xAC00) // 588))
            jamo_list.append(chr(0x1161 + ((ord(char) - 0xAC00) % 588) // 28))
            if (ord(char) - 0xAC00) % 28 != 0:
                jamo_list.append(chr(0x11A8 + (ord(char) - 0xAC00) % 28 - 1))
    return jamo_list


def normalize_space(text: str) -> str:
    """Normalizes spaces in text.
    The list of characters to be normalized is as follows:
    - U+0009 (tab)
    - U+000A (line feed)
    - U+000B (vertical tab)
    - U+000C (form feed)
    - U+000D (carriage return)
    - U+0020 (space)
    - U+0085 (next line)
    - U+00A0 (no-break space)
    - U+1680 (ogham space mark)
    - U+2000 (en quad)
    - U+2001 (em quad)
    - U+2002 (en space)
    - U+2003 (em space)
    - U+2004 (three-per-em space)
    - U+2005 (four-per-em space)
    - U+2006 (six-per-em space)
    - U+2007 (figure space)
    - U+2008 (punctuation space)
    - U+2009 (thin space)
    - U+200A (hair space)
    - U+2028 (line separator)
    - U+2029 (paragraph separator)
    - U+202F (narrow no-break space)
    - U+205F (medium mathematical space)
    - U+3000 (ideographic space)

    The list of characters to be removed is as follows:
    - U+200B (zero width space)
    - U+200C (zero width non-joiner)
    - U+200D (zero width joiner)
    - U+FEFF (zero width no-break space)

    Args:
        text (str): Text to normalize.

    Returns:
        str: Normalized text."""

    # Normalize spaces
    # Turns out python's str.split() does this already
    text = " ".join(text.split())

    # Remove characters
    text = text.replace("\u200B", "")
    text = text.replace("\u200C", "")
    text = text.replace("\u200D", "")
    text = text.replace("\uFEFF", "")

    return text


def remove_artifacts(text: str) -> str:
    """Removes artifacts from text.
    The list of artifacts to be removed is as follows:

    Excel bugs:
    - '_x000D_': Excel line break

    Args:
        text (str): Text to remove artifacts from.

    Returns:
        str: Text without artifacts."""

    text = text.replace("_x000D_", "")
    return text


def encode_text(text: str) -> np.ndarray:
    """Converts text into corresponding integer values.
    Handles Hangul & ASCII characters.
    Splits Hangul into individual characters.

    Args:
        text (str): Text to convert.

    Returns:
        np.ndarray: Shape: (N,)"""

    text = remove_artifacts(text)
    text = normalize_space(text)
    text = g2p(text)
    split_text = split_hangul_jamo(text)
    code = [CHAR_TO_CODE[char] for char in split_text]

    return np.array(code, dtype=np.int64)
