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

for path, package in (
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
):
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)

# Negations carry meaning in a query: "not a laptop" must not become "laptop".
_NEGATIONS = {"no", "nor", "not", "n't"}
_STOPWORDS = set(stopwords.words("english")) - _NEGATIONS
_LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in _STOPWORDS and t not in string.punctuation]
    # Lowercase first: WordNet only knows lowercase forms, so "Headphones" or a
    # sentence-initial "Designed" would otherwise pass through unlemmatised.
    tokens = [_LEMMATIZER.lemmatize(t.lower()) for t in tokens]
    return " ".join(tokens)
