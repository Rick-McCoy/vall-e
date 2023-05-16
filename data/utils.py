from pathlib import Path

import pandas as pd
from pandas import Series


def load_metadata(csv_path: Path) -> tuple["Series[str]", "Series[str]", "Series[str]"]:
    """Loads metadata from a CSV file.
    The CSV file contains three columns: speaker, text, and codec_path.
    The speaker column is used as the speaker list.
    The text column is used as the text list.
    The codec_path column is used as the codec path list.

    Args:
        csv_path (Path): Path to CSV file.

    Returns:
        speaker_list (Series[str]): Series of speakers.
        text_list (Series[str]): Series of texts.
        codec_path_list (Series[str]): Series of codec paths."""

    df = pd.read_csv(csv_path, index_col=False, dtype=str)
    speaker_list = df["speaker"]
    text_list = df["text"]
    codec_path_list = df["codec_path"]
    return speaker_list, text_list, codec_path_list
