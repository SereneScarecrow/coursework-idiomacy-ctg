"""
Module that creates dataset
"""

from idiomacy.metrics.text import normalize_text, tokenize, count_words
import pandas as pd

def normalize_entries(df, columns=["len_original", "len_literal"]):
    """
    Normalizes all text entries in dataset.
    """
    df["len_original"] = df["len_original"].apply(normalize_text)
    df["len_literal"] = df["len_literal"].apply(normalize_text)
    return df

def process_idioms():
    """
    """
    pass

def count_dataset_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all dataset metrics
    """
    df["len_original"] = df["original_text"].apply(count_words)
    df["len_literal"] = df["literal_text"].apply(count_words)
    df["len_idioms"] = df[""]
    df["TITR"] = df["len_idioms"] / df ["len_literal"]
    df["titr_level"] = 
    df["idiom_count"] = 
    df["len_ratio"] = df["len_original"] / df["len_literal"]

    return df

def main():
    pass
 
if __name__ == "__main__":
    main()