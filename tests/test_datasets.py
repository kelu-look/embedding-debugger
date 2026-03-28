"""Tests for demo datasets."""

import pytest
from demo.datasets import (
    load_faq,
    load_order_blind_pairs,
    load_news,
    load_nli,
    load_products,
    load_dataset,
    list_datasets,
    DATASETS,
)


def test_all_datasets_load():
    for name in DATASETS:
        texts, meta = load_dataset(name)
        assert isinstance(texts, list)
        assert len(texts) > 0
        assert isinstance(meta, dict)


def test_faq_pairs_aligned():
    texts, meta = load_faq()
    assert len(meta["questions"]) == len(meta["answers"])


def test_order_blind_pairs():
    texts, meta = load_order_blind_pairs()
    assert len(meta["originals"]) == len(meta["perturbed"])
    assert len(texts) == 2 * len(meta["originals"])


def test_news_categories_aligned():
    texts, meta = load_news()
    assert len(texts) == len(meta["categories"])


def test_nli_labels():
    _, meta = load_nli()
    valid = {"entailment", "contradiction", "neutral"}
    assert all(l in valid for l in meta["labels"])


def test_list_datasets():
    df = list_datasets()
    assert len(df) == len(DATASETS)
    assert "name" in df.columns
    assert "n_texts" in df.columns


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_dataset("does_not_exist")
