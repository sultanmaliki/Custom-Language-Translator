import pytest

import data_preprocessing as dp


@pytest.fixture(autouse=True)
def no_variants(monkeypatch):
    """Keep tests independent of whatever the user puts in data/spelling_variants.tsv."""
    monkeypatch.setattr(dp, "load_variants", lambda *a, **k: {})


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_read_pairs_normalises_dedupes_and_skips_comments(tmp_path):
    f = write(tmp_path / "pairs.tsv", (
        "﻿# a comment\n"
        "english\tnawayathi\tsource\n"
        "\n"
        "Hello  there\tHELLO thar\tamma\n"
        "Hello there\thello thar\t\n"          # same after normalisation -> duplicate
        "Good night\tSooo nite\t\n"
    ))
    pairs, report = dp.read_pairs(f)
    assert [(p.english, p.nawayathi) for p in pairs] == [("Hello there", "hello thar"), ("Good night", "soo nite")]
    assert pairs[0].source == "amma"
    assert report.rows_read == 3 and report.duplicates == 1 and not report.errors


def test_read_pairs_reports_line_numbers_for_rows_without_english(tmp_path):
    f = write(tmp_path / "pairs.tsv", "english\tnawayathi\nfine\tok\n\tonly nawayathi\n\t\tonly a source\n")
    _, report = dp.read_pairs(f)
    assert len(report.errors) == 2
    assert "line 3" in report.errors[0] and "line 4" in report.errors[1]


def test_english_only_rows_are_pending_not_errors(tmp_path):
    f = write(tmp_path / "pairs.tsv", (
        "english\tnawayathi\tsource\n"
        "Hello\t\tseed:greetings\n"
        "Good night\t   \t\n"
        "Only one column\n"
        "How are you\ttranslated text\t\n"
    ))
    pairs, report = dp.read_pairs(f)
    assert [p.english for p in pairs] == ["How are you"]
    assert report.pending == 3 and report.rows_read == 4 and not report.errors
    assert dp.corpus_size(f) == 1          # only translated rows count


def test_a_pending_row_does_not_block_adding_the_same_english_sentence(tmp_path):
    f = write(tmp_path / "pairs.tsv", "english\tnawayathi\tsource\nHello there\t\tseed:greetings\n")
    assert dp.append_pair("Hello there", "translated text", path=f) == "added"
    pairs, report = dp.read_pairs(f)
    assert len(pairs) == 1 and report.pending == 1


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16"])
def test_file_not_saved_as_utf8_gives_a_clear_error(tmp_path, encoding):
    f = tmp_path / "pairs.tsv"
    f.write_bytes("english\tnawayathi\ncafé\tx\n".encode(encoding))
    with pytest.raises(ValueError, match="UTF-8"):
        dp.read_pairs(f)
    assert dp.corpus_size(f) == 0


def test_read_pairs_requires_the_header_columns(tmp_path):
    f = write(tmp_path / "pairs.tsv", "eng\tnwy\nhello\thi\n")
    with pytest.raises(ValueError, match="header"):
        dp.read_pairs(f)


def test_read_pairs_missing_file_explains_the_format(tmp_path):
    with pytest.raises(FileNotFoundError, match="header row"):
        dp.read_pairs(tmp_path / "nope.tsv")


def test_warnings_for_non_roman_text_extra_columns_and_untranslated_copies(tmp_path):
    f = write(tmp_path / "pairs.tsv", (
        "english\tnawayathi\n"
        "hello\tनमस्ते\n"      # Devanagari
        "cat\tcat\n"                                        # identical both sides
        "tab\tin\tside\tsentence\n"                         # extra columns
    ))
    _, report = dp.read_pairs(f)
    text = " | ".join(report.warnings)
    assert "non-Roman" in text and "identical" in text and "more columns" in text
    assert not report.errors


def make_pairs(n, variants_of_first=0):
    pairs = [dp.Pair(f"sentence {i}", f"vaakya {i}") for i in range(n)]
    pairs += [dp.Pair("sentence 0", f"vaakya 0 alt{k}") for k in range(variants_of_first)]
    return pairs


def test_split_has_no_english_leakage_and_keeps_variants_together():
    pairs = make_pairs(40, variants_of_first=3)
    train, val, test = dp.split_pairs(pairs, seed=1)
    sets = [{p.english.lower() for p in part} for part in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(train) + len(val) + len(test) == len(pairs)
    assert len(val) >= 2 and len(test) >= 2 and len(train) > len(val)


def test_split_is_deterministic_per_seed_and_differs_across_seeds():
    pairs = make_pairs(60)
    assert dp.split_pairs(pairs, seed=3) == dp.split_pairs(pairs, seed=3)
    assert dp.split_pairs(pairs, seed=3)[2] != dp.split_pairs(pairs, seed=4)[2]


def test_split_refuses_tiny_corpora():
    with pytest.raises(ValueError, match="at least"):
        dp.split_pairs(make_pairs(5))


def test_split_round_trips_through_files(tmp_path):
    train, _, _ = dp.split_pairs(make_pairs(40), seed=1)
    dp.write_split(tmp_path / "train.tsv", train)
    assert dp.read_split(tmp_path / "train.tsv") == [dp.Pair(p.english, p.nawayathi) for p in train]


def test_append_creates_file_with_header_and_detects_duplicates(tmp_path):
    f = tmp_path / "data" / "pairs.tsv"
    assert dp.append_pair("Hello there", "hello thar", "amma", path=f) == "added"
    assert f.read_text(encoding="utf-8").splitlines()[0] == "english\tnawayathi\tsource"
    assert dp.append_pair("  hello   there ", "HELLO thar", path=f) == "duplicate"   # same after normalising
    assert dp.append_pair("Hello there", "hello thar (other spelling)", path=f) == "added"
    assert dp.corpus_size(f) == 2


def test_append_sanitises_tabs_and_newlines_so_the_tsv_stays_valid(tmp_path):
    f = tmp_path / "pairs.tsv"
    dp.append_pair("two\tparts\nhere", "do\tbhaag", path=f)
    pairs, report = dp.read_pairs(f)
    assert [(p.english, p.nawayathi) for p in pairs] == [("two parts here", "do bhaag")]
    assert not report.warnings


def test_append_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        dp.append_pair("hello", "   ", path=tmp_path / "pairs.tsv")


def test_append_repairs_a_missing_final_newline(tmp_path):
    f = write(tmp_path / "pairs.tsv", "english\tnawayathi\tsource\nfirst\tone\t")  # no trailing newline
    dp.append_pair("second", "two", path=f)
    pairs, _ = dp.read_pairs(f)
    assert [p.english for p in pairs] == ["first", "second"]


def test_append_refuses_a_file_with_a_broken_header(tmp_path):
    f = write(tmp_path / "pairs.tsv", "wrong\theader\n")
    with pytest.raises(ValueError, match="header"):
        dp.append_pair("a", "b", path=f)
    assert f.read_text(encoding="utf-8") == "wrong\theader\n"


def test_corpus_size_is_zero_for_missing_file(tmp_path):
    assert dp.corpus_size(tmp_path / "nope.tsv") == 0
