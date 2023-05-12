import numpy as np

# Special tokens
char_to_code = {
    "<pad>": 0,
    "<unk>": 1,
    "<eos>": 2,
}
# ASCII
char_to_code.update({chr(i): i + 3 - 0 for i in range(128)})
# Hangul Jamo (Initial)
char_to_code.update({chr(i): i + 131 - 0x1100 for i in range(0x1100, 0x1113)})
# Hangul Jamo (Medial)
char_to_code.update({chr(i): i + 144 - 0x1161 for i in range(0x1161, 0x1176)})
# Hangul Jamo (Final)
char_to_code.update({chr(i): i + 157 - 0x11A8 for i in range(0x11A8, 0x11C3)})

VOCAB_SIZE = len(char_to_code)


def split_hangul_jamo(text: str) -> list[str]:
    """Splits Hangul into individual characters.

    Args:
        text (str): Text to split.

    Returns:
        List[str]: Split text."""
    raise NotImplementedError


def encode_text(text: str) -> np.ndarray:
    """Converts text into corresponding integer values.
    Handles Hangul & ASCII characters.
    Splits Hangul into individual characters.

    Args:
        text (str): Text to convert.

    Returns:
        np.ndarray: Shape: (N,)"""

    code = []
    for char in split_hangul_jamo(text):
        code.append(char_to_code[char])

    return np.array(code, dtype=np.int64)
