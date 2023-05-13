import numpy as np

char_to_code = {}
# ASCII (128 characters)
char_to_code.update({chr(i): i for i in range(128)})
# Hangul Jamo (Initial, 19 characters)
char_to_code.update({chr(i): i + 128 - 0x1100 for i in range(0x1100, 0x1113)})
# Hangul Jamo (Medial, 21 characters)
char_to_code.update({chr(i): i + 147 - 0x1161 for i in range(0x1161, 0x1176)})
# Hangul Jamo (Final, 28 characters)
char_to_code.update({chr(i): i + 168 - 0x11A8 for i in range(0x11A8, 0x11C3)})

VOCAB_SIZE = len(char_to_code)


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
        else:
            jamo_list.append(char)
    return jamo_list


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
