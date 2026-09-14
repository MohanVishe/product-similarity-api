"""Text normalisation applied identically at index time and query time.

Both sides must use this - embedding a raw query against preprocessed
documents silently degrades retrieval, and it is the kind of bug that
produces plausible-looking bad results rather than an error.
"""
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for resource in ("punkt", "stopwords", "wordnet"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in _STOPWORDS and t not in string.punctuation]
    tokens = [_LEMMATIZER.lemmatize(t).lower() for t in tokens]
    return " ".join(tokens)
