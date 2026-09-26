"""Validate the corpus and split it into train / val / test.

    python data_preprocessing.py

Reads data/pairs.tsv, a tab-separated file with a header row:

    english <TAB> nawayathi <TAB> source        (source is optional free text)

Every problem is reported with its line number. Errors stop the run so that
nothing is silently dropped from a corpus you built by hand; warnings don't.
The split is grouped by English sentence, so the same English sentence (with
several Nawayathi spellings) never lands in both train and test.
"""
import argparse
import csv
import random
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

import config
from text_normalization import (
    clean_raw, load_variants, normalize_english, normalize_nawayathi, use_utf8_console,
)

HEADER = ["english", "nawayathi", "source"]
_NON_ROMAN = re.compile(r"[؀-ۿݐ-ݿऀ-ॿಀ-೿]")  # Arabic, Devanagari, Kannada
_LONG_WORDS = 60
_append_lock = threading.Lock()


@dataclass
class Pair:
    english: str
    nawayathi: str
    source: str = ""


@dataclass
class Report:
    rows_read: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _iter_rows(path: Path):
    """Yield (line_number, {column: value}) for every data row after the header."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = None
        for row in reader:
            if not row or not "".join(row).strip() or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [h.strip().lower() for h in row]
                missing = {"english", "nawayathi"} - set(header)
                if missing:
                    raise ValueError(
                        f"{Path(path).name}: header row must contain 'english' and 'nawayathi' "
                        f"columns, found {header}"
                    )
                continue
            rec = dict(zip(header, row))
            rec["_extra"] = len(row) > len(header)
            yield reader.line_num, rec


def read_pairs(path: Path = config.PAIRS_FILE) -> tuple[list[Pair], Report]:
    """Load, normalise, validate and de-duplicate the corpus."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with a header row: english<TAB>nawayathi<TAB>source")
    variants = load_variants()
    report, pairs, seen = Report(), [], set()

    for line_no, rec in _iter_rows(path):
        report.rows_read += 1
        english = normalize_english(rec.get("english", ""))
        nawayathi = normalize_nawayathi(rec.get("nawayathi", ""), variants)
        if not english or not nawayathi:
            report.errors.append(f"line {line_no}: needs both an English and a Nawayathi sentence")
            continue
        if rec["_extra"]:
            report.warnings.append(f"line {line_no}: more columns than the header (is there a stray tab in a sentence?)")
        if _NON_ROMAN.search(nawayathi):
            report.warnings.append(f"line {line_no}: Nawayathi text contains non-Roman characters: {nawayathi!r}")
        if max(len(english.split()), len(nawayathi.split())) > _LONG_WORDS:
            report.warnings.append(f"line {line_no}: very long sentence (>{_LONG_WORDS} words); consider splitting it")
        if english.lower() == nawayathi.lower():
            report.warnings.append(f"line {line_no}: both sides are identical: {english!r}")

        key = (english.lower(), nawayathi)
        if key in seen:
            report.duplicates += 1
            continue
        seen.add(key)
        pairs.append(Pair(english, nawayathi, rec.get("source", "").strip()))
    return pairs, report


def split_pairs(
    pairs: list[Pair],
    val_fraction: float = config.VAL_FRACTION,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.SEED,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    """Split by English sentence so no sentence leaks between train and test."""
    n = len(pairs)
    if n < config.MIN_PAIRS:
        raise ValueError(f"Only {n} usable pairs; need at least {config.MIN_PAIRS} to make a train/val/test split.")

    groups: dict[str, list[int]] = {}
    for i, p in enumerate(pairs):
        groups.setdefault(p.english.lower(), []).append(i)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    n_test = max(2, round(n * test_fraction))
    n_val = max(2, round(n * val_fraction))
    test_idx, val_idx, train_idx = [], [], []
    for key in keys:
        if len(test_idx) < n_test:
            test_idx += groups[key]
        elif len(val_idx) < n_val:
            val_idx += groups[key]
        else:
            train_idx += groups[key]
    if not train_idx:
        raise ValueError("Not enough distinct English sentences to leave any for training.")

    # keep original file order inside each split so the output is easy to diff/inspect
    return tuple([pairs[i] for i in sorted(idx)] for idx in (train_idx, val_idx, test_idx))


def write_split(path: Path, pairs: list[Pair]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        f.write("english\tnawayathi\n")
        for p in pairs:
            f.write(f"{p.english}\t{p.nawayathi}\n")


def read_split(path: Path) -> list[Pair]:
    return [Pair(rec["english"], rec["nawayathi"]) for _, rec in _iter_rows(path)]


def corpus_size(path: Path = config.PAIRS_FILE) -> int:
    """Number of data rows in the corpus (tolerant: 0 if the file is absent or unreadable)."""
    try:
        return sum(1 for _ in _iter_rows(path))
    except (OSError, ValueError):
        return 0


def append_pair(english: str, nawayathi: str, source: str = "", path: Path = config.PAIRS_FILE) -> str:
    """Add one pair to the corpus. Returns 'added' or 'duplicate'; raises ValueError on empty input."""
    path = Path(path)
    english, nawayathi, source = clean_raw(english), clean_raw(nawayathi), clean_raw(source)
    if not english or not nawayathi:
        raise ValueError("Both an English and a Nawayathi sentence are required.")

    variants = load_variants()
    key = (normalize_english(english).lower(), normalize_nawayathi(nawayathi, variants))
    with _append_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size == 0:
            path.write_text("\t".join(HEADER) + "\n", encoding="utf-8", newline="")
        # _iter_rows raises ValueError on a malformed header, so we never append to a file we can't read
        for _, rec in _iter_rows(path):
            existing = (
                normalize_english(rec.get("english", "")).lower(),
                normalize_nawayathi(rec.get("nawayathi", ""), variants),
            )
            if existing == key:
                return "duplicate"
        with path.open("rb+") as f:  # make sure the last line is terminated before appending
            f.seek(0, 2)
            if f.tell() > 0:
                f.seek(-1, 2)
                if f.read(1) not in (b"\n", b"\r"):
                    f.write(b"\n")
        with path.open("a", encoding="utf-8", newline="") as f:
            f.write(f"{english}\t{nawayathi}\t{source}\n")
    return "added"


def main() -> None:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, default=config.PAIRS_FILE)
    parser.add_argument("--out", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    pairs, report = read_pairs(args.pairs)
    for w in report.warnings:
        print(f"warning: {w}")
    if report.errors:
        for e in report.errors:
            print(f"error: {e}")
        sys.exit(f"\n{len(report.errors)} error(s) in {args.pairs.name}. Fix them and run again.")

    print(f"Read {report.rows_read} rows -> {len(pairs)} unique pairs ({report.duplicates} duplicates skipped).")
    try:
        train, val, test = split_pairs(pairs, seed=args.seed)
    except ValueError as e:
        sys.exit(str(e))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        write_split(args.out / f"{name}.tsv", part)
    print(f"Split -> train {len(train)}, val {len(val)}, test {len(test)}  (written to {args.out})")

    multi = sum(1 for n in _variant_counts(pairs).values() if n > 1)
    if multi:
        print(f"{multi} English sentence(s) have more than one Nawayathi version (fine: they stay in the same split).")
    if len(train) < 500:
        print(
            f"Note: {len(train)} training pairs is a very small corpus. The pipeline will run, but expect the "
            "model to mostly memorise; quality improves as you add more pairs."
        )


def _variant_counts(pairs: list[Pair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for p in pairs:
        counts[p.english.lower()] = counts.get(p.english.lower(), 0) + 1
    return counts


if __name__ == "__main__":
    main()
