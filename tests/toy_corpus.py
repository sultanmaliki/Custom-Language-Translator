"""A tiny INVENTED language ("Toylang") used only to test the pipeline end to end.

This is NOT Nawayathi and must never be written into data/pairs.tsv. The point is
to have a corpus whose answer we know: if the pipeline is wired correctly, a model
trained on it should quickly learn the word mapping and the (deliberately different)
word order, which a broken pipeline can't do.
"""
import random

# English word -> made-up Roman-script word
NOUNS = {"dog": "kuvo", "cat": "miru", "child": "tasel", "bird": "pilo", "house": "romba", "river": "sunet",
         "tree": "gorva", "boy": "danik", "girl": "vesha", "book": "lopi"}
VERBS = {"sees": "nara", "hears": "tuli", "wants": "beko", "finds": "zipa", "likes": "hamo"}
ADJECTIVES = {"big": "shano", "small": "pinu", "red": "kelto", "old": "marbi", "happy": "fuda"}


def make_pairs(n: int, seed: int = 0) -> list[tuple[str, str]]:
    """English 'the ADJ NOUN VERB the NOUN' -> Toylang 'NOUN ADJ NOUN VERB' (article dropped, verb last)."""
    rng = random.Random(seed)
    pairs: set[tuple[str, str]] = set()
    attempts = 0
    while len(pairs) < n and attempts < n * 50:
        attempts += 1
        adj, subj, verb, obj = (rng.choice(list(d)) for d in (ADJECTIVES, NOUNS, VERBS, NOUNS))
        english = f"the {adj} {subj} {verb} the {obj}"
        toy = f"{NOUNS[subj]} {ADJECTIVES[adj]} {NOUNS[obj]} {VERBS[verb]}"
        pairs.add((english, toy))
    return sorted(pairs)


def write_corpus(path, n: int = 260, seed: int = 0) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("english\tnawayathi\tsource\n")
        for english, toy in make_pairs(n, seed):
            f.write(f"{english}\t{toy}\ttoy\n")
