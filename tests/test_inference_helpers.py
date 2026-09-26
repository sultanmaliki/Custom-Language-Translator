import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from inference import require_language_tags, split_sentences  # noqa: E402


class FakeTokenizer:
    """Mimics NLLB: unknown tags come back as the <unk> id instead of raising."""
    unk_token_id = 3
    vocab = {"eng_Latn": 256047, "mar_Deva": 256116}

    def convert_tokens_to_ids(self, tag):
        return self.vocab.get(tag, self.unk_token_id)


def test_known_language_tags_pass():
    require_language_tags(FakeTokenizer(), ("eng_Latn", "mar_Deva"), "fake-model")


def test_unknown_language_tag_fails_loudly_instead_of_becoming_unk():
    # 'gom_Deva' (Konkani) is NOT in Meta's NLLB-200; using it would silently train on <unk>
    with pytest.raises(ValueError, match="gom_Deva"):
        require_language_tags(FakeTokenizer(), ("eng_Latn", "gom_Deva"), "fake-model")


def test_split_sentences():
    assert split_sentences("One. Two!  Three? Four") == ["One.", "Two!", "Three?", "Four"]
    assert split_sentences("   ") == []
    assert split_sentences("no terminator") == ["no terminator"]
    assert split_sentences("3.5 is a number") == ["3.5 is a number"]   # no whitespace after the dot: not a boundary
