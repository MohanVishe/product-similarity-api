"""Exclusion parsing: which "not X" phrases become filters, and which do not."""
import pandas as pd
import pytest

from conftest import ROOT
from exclusions import Product, match_products, parse_query

PRODUCTS = [
    Product(row["Product Name"], row["Category"])
    for _, row in pd.read_csv(ROOT / "data" / "Generated_Product_Data.csv").iterrows()
]
ELECTRONICS = ["Laptop", "Smartphone", "Wireless Headphones", "Portable Speaker"]


@pytest.mark.parametrize(
    "query, excluded, text",
    [
        ("not a laptop, a phone", ["Laptop"], "a phone"),
        ("fitness equipment that is not a yoga mat", ["Yoga Mat"], "fitness equipment that is"),
        ("kitchen things without the coffee maker", ["Coffee Maker"], "kitchen things"),
        ("music gear except headphones", ["Wireless Headphones"], "music gear"),
        ("music gear except for headphones", ["Wireless Headphones"], "music gear"),
        ("fitness stuff excluding dumbbells", ["Dumbbell Set"], "fitness stuff"),
        ("accessories other than sunglasses", ["Sunglasses"], "accessories"),
        ("travel gear, no backpack", ["Backpack"], "travel gear"),
        ("accessories that aren't watches", ["Watch"], "accessories that are"),
        ("gadgets but not electronics", ELECTRONICS, "gadgets but"),
        ("no laptop, no watch, something stylish", ["Laptop", "Watch"], "something stylish"),
    ],
)
def test_exclusions_become_filters_and_leave_the_rest(query, excluded, text):
    parsed = parse_query(query, PRODUCTS)
    assert parsed.excluded == excluded
    assert parsed.text == text


@pytest.mark.parametrize(
    "query",
    [
        "a watch that is not too expensive",          # hedge: a qualifier, not an exclusion
        "not just a laptop, I also need a chair",     # "not just X" means "X and more"
        "headphones that are not wired",              # "wired" names no product
        "a yoga mat with no slipping",
        "electronics but not a phone",                # no synonyms: "phone" is not "Smartphone"
        "not wired headphones",                       # every word must match: nothing is "wired"
    ],
)
def test_phrases_that_name_no_product_change_nothing(query):
    parsed = parse_query(query, PRODUCTS)
    assert parsed.excluded == []
    assert parsed.text == query


def test_bare_negation_keeps_the_typed_text_for_ranking():
    parsed = parse_query("not a laptop", PRODUCTS)
    assert parsed.excluded == ["Laptop"]
    assert parsed.text == "not a laptop"


def test_matching_needs_every_word_in_the_name_or_category():
    assert match_products("a yoga mat", PRODUCTS) == ["Yoga Mat"]
    assert match_products("shoes", PRODUCTS) == ["Running Shoes"]
    assert match_products("kitchen", PRODUCTS) == ["Coffee Maker"]  # "Kitchen Appliances"
    assert match_products("sets", PRODUCTS) == ["Cookware Set", "Dumbbell Set", "Comforter Set"]
    assert match_products("cheap laptop", PRODUCTS) == []
    assert match_products("too expensive", PRODUCTS) == []
