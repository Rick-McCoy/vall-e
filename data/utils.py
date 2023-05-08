from pathlib import Path

import pandas as pd
from pandas import Series


def load_metadata(csv_path: Path) -> tuple["Series[str]", "Series[str]", "Series[str]"]:
    """Loads metadata from a CSV file.
    The CSV file contains four columns: speaker, emotion, text, and codec_path.
    The emotion column is combined with the speaker column to create the speaker list.
    The text column is used as the text list.
    The codec_path column is used as the codec path list.

    Args:
        csv_path (Path): Path to CSV file.

    Returns:
        speaker_list (Series[str]): Series of speakers.
        text_list (Series[str]): Series of texts.
        codec_path_list (Series[str]): Series of codec paths."""

    df = pd.read_csv(csv_path)
    speaker_list = df["speaker"].astype(str) + "-" + df["emotion"].astype(str)
    text_list = df["text"].astype(str)
    codec_path_list = df["codec_path"].astype(str)
    return speaker_list, text_list, codec_path_list
