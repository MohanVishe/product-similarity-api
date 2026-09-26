"""Turn "not X" / "without X" / "except X" in a query into an exclusion filter.

Embedding models do not act on negation: "not a laptop" embeds close to
"laptop", so the laptop ranks first. Instead of embedding the negated phrase,
this module finds it, resolves it to catalogue products by name or category,
drops those products from the candidate set, and removes the phrase from the
text that is embedded and keyword-matched.

Rules (deliberately simple and predictable):

- Triggers: "not", "n't" (isn't, aren't, won't...), "no", "without",
  "except" / "except for", "excluding", "other than".
- The phrase X runs from the trigger to the next punctuation mark or clause
  word ("but", "and", "or", "that", "which", "with", "for", ...).
- X is normalised with the same `preprocess_text` as the index, and it excludes
  a product only if EVERY remaining word of X is a word of that product's name
  or category. "a yoga mat" -> Yoga Mat; "shoes" -> Running Shoes;
  "electronics" -> the whole Electronics category; "wired headphones" -> nothing
  (no product is called "wired").
- Hedges are not exclusions: when X starts with "too", "very", "just", "only",
  "so", ... ("not too expensive", "not just a phone") nothing is excluded.
- A phrase that matches no product changes nothing: the query text is left
  exactly as typed, so "headphones that are not wired" still searches for it.

What it does not do: synonyms ("phone" does not match "Smartphone"), scope
("not a laptop but a laptop bag"), or double negatives. Those phrases fall
through to the unchanged query text.
"""
import re
from dataclasses import dataclass, field

from preprocess import preprocess_text

_TRIGGER = re.compile(
    r"\b(?:other\s+than|except(?:\s+for)?|excluding|without|not|no)\b|n['’]t\b",
    re.IGNORECASE,
)
_BOUNDARY = re.compile(
    r"[,.;:!?()]|\b(?:but|and|or|nor|that|which|who|with|for|so|because|though|"
    r"although|while|instead|rather|please)\b",
    re.IGNORECASE,
)
# "not too expensive", "not just a phone", "not that fancy": a qualifier, not an exclusion.
_HEDGES = {
    "too", "very", "so", "just", "only", "that", "as", "quite", "really", "overly",
    "necessarily", "exactly", "much", "even", "entirely", "always", "sure",
}


@dataclass(frozen=True)
class Product:
    name: str
    category: str

    @property
    def words(self) -> frozenset:
        return frozenset(preprocess_text(f"{self.name} {self.category}").split())


@dataclass
class ParsedQuery:
    original: str
    text: str                                      # what gets embedded / keyword-matched
    excluded: list = field(default_factory=list)   # product names ruled out, catalogue order
    phrases: list = field(default_factory=list)    # the negated phrases that matched products


def _phrase_after(query: str, start: int) -> tuple[str, int]:
    """The negated phrase starting at `start`, and where it ends."""
    boundary = _BOUNDARY.search(query, start)
    end = boundary.start() if boundary else len(query)
    return query[start:end], end


def match_products(phrase: str, products) -> list[str]:
    """Names of the products whose name/category words cover every word of `phrase`."""
    first = phrase.strip().split(maxsplit=1)
    if first and first[0].lower().strip("'’") in _HEDGES:
        return []
    words = set(preprocess_text(phrase).split())
    if not words:
        return []
    return [p.name for p in products if words <= p.words]


def parse_query(query: str, products) -> ParsedQuery:
    """Find negated phrases that name catalogue products; strip them from the text."""
    products = list(products)
    excluded, phrases, cut = [], [], []
    for trigger in _TRIGGER.finditer(query):
        if cut and trigger.start() < cut[-1][1]:
            continue  # inside a phrase already cut
        phrase, end = _phrase_after(query, trigger.end())
        names = match_products(phrase, products)
        if not names:
            continue
        phrases.append(phrase.strip())
        cut.append((trigger.start(), end))
        excluded += [n for n in names if n not in excluded]

    if not cut:
        return ParsedQuery(original=query, text=query)

    pieces, position = [], 0
    for start, end in cut:
        pieces.append(query[position:start])
        position = end
    pieces.append(query[position:])
    text = re.sub(r"\s+", " ", "".join(pieces)).strip(" ,;")
    # "not a laptop" leaves nothing to search with: keep the typed text for ranking
    # (the excluded products are still filtered out).
    if not preprocess_text(text).split():
        text = query
    order = {p.name: i for i, p in enumerate(products)}
    return ParsedQuery(
        original=query, text=text, excluded=sorted(excluded, key=order.get), phrases=phrases
    )
