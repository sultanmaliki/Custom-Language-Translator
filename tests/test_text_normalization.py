from text_normalization import clean_raw, load_variants, normalize_english, normalize_nawayathi


def test_english_cleanup_keeps_case_but_fixes_spaces_quotes_and_invisibles():
    assert normalize_english("  Hello​   “world” \n") == 'Hello "world"'


def test_nawayathi_is_lowercased_and_stretched_letters_collapsed():
    assert normalize_nawayathi("SoooO  Good") == "soo good"
    assert normalize_nawayathi("Aa") == "aa"  # a legitimate double letter survives


def test_variants_replace_whole_words_only_and_case_insensitively():
    variants = {"foo": "baz"}
    assert normalize_nawayathi("Foo bar, FOO! foobar", variants) == "baz bar, baz! foobar"


def test_variants_handle_apostrophes():
    assert normalize_nawayathi("Fo’o", {"fo'o": "foo"}) == "foo"


def test_load_variants_ignores_comments_blank_lines_and_missing_file(tmp_path):
    f = tmp_path / "v.tsv"
    f.write_text("# comment\n\nOne\tTwo\nbroken line\n", encoding="utf-8")
    assert load_variants(f) == {"one": "two"}
    assert load_variants(tmp_path / "missing.tsv") == {}


def test_clean_raw_only_touches_whitespace():
    assert clean_raw("  Keep\tCASE\n and  Stuff ") == "Keep CASE and Stuff"
