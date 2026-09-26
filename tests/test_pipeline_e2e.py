"""End to end on a toy language: preprocess -> train -> evaluate -> translate.

Slow (loads the real NLLB model and trains on the GPU, ~3 minutes), so it only runs on request:

    $env:RUN_SLOW = "1"; pytest tests/test_pipeline_e2e.py -s

The toy language (see toy_corpus.py) has a known answer, so a pipeline that is wired wrongly
(bad language tags, shifted labels, broken generation) can't score well on it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import toy_corpus

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parent.parent


def run(script: str, *args: str) -> str:
    result = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 to run the end-to-end test")
def test_toy_language_is_learned_in_both_directions(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (CPU training would take far too long)")

    pairs, processed, adapter = tmp_path / "pairs.tsv", tmp_path / "processed", tmp_path / "adapter"
    toy_corpus.write_corpus(pairs, n=260)

    run("data_preprocessing.py", "--pairs", str(pairs), "--out", str(processed))
    run("train_model.py", "--data-dir", str(processed), "--output-dir", str(adapter), "--epochs", "12")
    run("evaluate.py", "--data-dir", str(processed), "--adapter-dir", str(adapter), "--samples", "0")

    scores = json.loads((adapter / "eval.json").read_text(encoding="utf-8"))
    for direction in ("en2nwy", "nwy2en"):
        assert scores[direction]["chrF++"] >= 90, f"{direction}: {scores[direction]}"

    # an unseen sentence, translated through the public API in both directions
    from inference import Translator
    translator = Translator(adapter)
    assert translator.translate("the old girl likes the tree", "en2nwy") == "vesha marbi gorva hamo"
    assert translator.translate("vesha marbi gorva hamo", "nwy2en") == "the old girl likes the tree"
