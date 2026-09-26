"""Score the trained model on the held-out test split, in both directions.

    python evaluate.py

chrF++ is the number to watch (0-100, higher is better); it works on characters, so
it copes with Nawayathi's variable spelling much better than BLEU does. With a small
test set every score is noisy, so read the sample translations too.
"""
import argparse
import json
from pathlib import Path

import sacrebleu

import config
from data_preprocessing import read_split
from inference import ModelNotTrainedError, Translator
from text_normalization import use_utf8_console


def score(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    return {
        "chrF++": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 1),
        "BLEU": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 1),
    }


def main():
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--samples", type=int, default=8, help="how many example translations to print per direction")
    args = parser.parse_args()

    test_file = args.data_dir / "test.tsv"
    if not test_file.exists():
        raise SystemExit(f"{test_file} not found. Run data_preprocessing.py first.")
    test = read_split(test_file)
    try:
        translator = Translator(args.adapter_dir)
    except ModelNotTrainedError as e:
        raise SystemExit(str(e))

    results = {}
    for direction in ("en2nwy", "nwy2en"):
        sources = [p.english if direction == "en2nwy" else p.nawayathi for p in test]
        references = [p.nawayathi if direction == "en2nwy" else p.english for p in test]
        hypotheses = translator.translate_batch(sources, direction, num_beams=args.beams)
        results[direction] = {**score(hypotheses, references), "sentences": len(test)}
        print(f"\n=== {direction}: {results[direction]} ===")
        for src, hyp, ref in list(zip(sources, hypotheses, references))[: args.samples]:
            print(f"  source:    {src}\n  model:     {hyp}\n  reference: {ref}\n")

    (args.adapter_dir / "eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Scores saved to {args.adapter_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
