from preprocess import preprocess_text


def test_negation_is_kept():
    assert preprocess_text("not a laptop") == "not laptop"


def test_contracted_negation_is_kept():
    tokens = preprocess_text("This isn't a phone").split()
    assert "n't" in tokens and "phone" in tokens


def test_no_and_nor_are_kept():
    tokens = preprocess_text("no cables, nor batteries").split()
    assert tokens[0] == "no" and "nor" in tokens


def test_other_stopwords_and_punctuation_are_dropped():
    assert preprocess_text("The shoes, for the runners!") == "shoe runner"


def test_lowercases_and_lemmatises():
    assert preprocess_text("Headphones") == "headphone"
