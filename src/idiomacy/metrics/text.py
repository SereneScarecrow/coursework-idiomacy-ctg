"""
Text processing module.
"""

import argparse
import re
from functools import lru_cache
from typing import List, Tuple

import spacy


HYPHENATED_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)+$")

def normalize_text(text: str) -> str:
    """
    Normalizes text:
        - removes line breaks inside paragraphs;
        - fixes broken spaces;
        - squashes scattered words;
    TODO: unifies quotation marks and hyphens.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.replace('\xa0', ' ')

    # squashes scattered words (longer than 5 letters)
    text = re.sub(r'\b(?:\w\s){4,}\w\b', lambda m: m.group(0).replace(' ', ''), text)

    # removes line breaks inside paragraphs
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = [' '.join(p.splitlines()) for p in paragraphs]
    normalized_text = '\n\n'.join(cleaned_paragraphs)
    
    # cleans up multiple spaces
    normalized_text = re.sub(r'[ \t]+', ' ', normalized_text)
    normalized_text = re.sub(r' +', ' ', normalized_text).strip()
    
    return normalized_text

@lru_cache(maxsize=None)
def get_nlp(model: str = "ru_core_news_sm"):
    nlp = spacy.load(model)
    previous = nlp.tokenizer.token_match

    def token_match(text):
        return HYPHENATED_WORD.match(text) or (previous(text) if previous else None)

    nlp.tokenizer.token_match = token_match
    return nlp

def tokenize(normalized_text: str, model: str = "ru_core_news_sm") -> list[str]:
    """
    Returns tokenized text. Both parts of a word with hyphen considered to be one word.
    """
    doc = get_nlp(model)(normalized_text)
    return [token.text for token in doc]

def count_words():
    pass

def sentences():
    """
    Divides text into sentences.
    """
    pass

def content_tokens():
    """
    Filters tokens, returns only meaningful tokens.
    """
    pass

def main():
    parser = argparse.ArgumentParser(description="Text processing")
    parser.add_argument("text", nargs="?", help="text in quotation marks")
    parser.add_argument("--file", help="path to the text file (UTF-8)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            raw_text = f.read()
    elif args.text:
        raw_text = args.text
    else:
        parser.error("pass the text as an argument or through --file")

    clean_text = normalize_text(raw_text)
    print(tokenize(clean_text))


if __name__ == "__main__":
    main()