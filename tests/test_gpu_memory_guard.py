import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import train_model as tm  # noqa: E402

NLLB_VOCAB = 256_206
GIB = 2**30


def test_estimate_matches_measurements_from_the_rtx_4060_laptop():
    # (batch size, target length, measured peak GiB minus the 1.18 GiB of resident weights)
    measured = [(8, 96, 3.46), (16, 32, 2.33), (32, 32, 4.57), (32, 64, 9.04)]
    for batch, length, gib in measured:
        assert tm.estimate_activation_gib(NLLB_VOCAB, batch, length) == pytest.approx(gib, abs=0.15)


def test_estimate_grows_with_batch_and_length():
    base = tm.estimate_activation_gib(NLLB_VOCAB, 8, 32)
    assert tm.estimate_activation_gib(NLLB_VOCAB, 16, 32) > base
    assert tm.estimate_activation_gib(NLLB_VOCAB, 8, 64) > base


def test_project_defaults_are_safe_on_this_gpu():
    # --batch-size 8 with --max-len 96 on ~5.8 GiB free (what is left after loading the model)
    assert tm.safe_batch_size(NLLB_VOCAB, 96, 5.76) >= 8


def test_safe_batch_size_shrinks_with_less_memory_and_never_hits_zero():
    assert tm.safe_batch_size(NLLB_VOCAB, 64, 3.0) < tm.safe_batch_size(NLLB_VOCAB, 64, 6.0)
    assert tm.safe_batch_size(NLLB_VOCAB, 512, 0.5) == 1


def fake_free_memory(monkeypatch, free_gib):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (int(free_gib * GIB), 8 * GIB))


def test_guard_stays_quiet_when_the_batch_fits(monkeypatch, capsys):
    fake_free_memory(monkeypatch, 5.8)
    tm.check_gpu_memory(NLLB_VOCAB, batch_size=8, accum=2, longest_target=48)
    out = capsys.readouterr().out
    assert "(OK)" in out and "WARNING" not in out


def test_guard_warns_and_suggests_a_smaller_batch_that_keeps_the_effective_batch(monkeypatch, capsys):
    fake_free_memory(monkeypatch, 5.8)
    tm.check_gpu_memory(NLLB_VOCAB, batch_size=32, accum=2, longest_target=96)
    out = capsys.readouterr().out
    assert "WARNING" in out and "spill" in out
    safe = tm.safe_batch_size(NLLB_VOCAB, 96, 5.8)
    assert f"--batch-size {safe} --accum {-(-32 * 2 // safe)}" in out   # ceil(64 / safe)


def test_collate_pads_to_a_multiple_and_keeps_decoder_inputs_aligned_with_labels():
    collate = tm.make_collate(pad_id=1, decoder_start_id=2, pad_multiple=8)
    batch = collate([
        {"input_ids": [10, 11, 2], "labels": [20, 21, 22, 2]},
        {"input_ids": [10] * 11, "labels": [20, 2]},
    ])
    assert batch["input_ids"].shape == (2, 16) and batch["labels"].shape == (2, 8)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1] + [0] * 13
    assert batch["labels"][0].tolist() == [20, 21, 22, 2, -100, -100, -100, -100]
    # the decoder reads: start token, then the labels shifted right, then padding
    assert batch["decoder_input_ids"][0].tolist() == [2, 20, 21, 22, 1, 1, 1, 1]
    assert batch["decoder_input_ids"][1].tolist() == [2, 20, 1, 1, 1, 1, 1, 1]


def test_collate_without_padding_multiple_uses_exact_lengths():
    batch = tm.make_collate(1, 2)([{"input_ids": [10, 11, 2], "labels": [20, 21, 22, 2]}])
    assert batch["input_ids"].shape == (1, 3) and batch["labels"].shape == (1, 4)


def test_allocator_fraction_leaves_headroom_below_the_free_vram():
    assert tm.allocator_fraction(6.94, 8.0) == pytest.approx(0.80, abs=0.005)   # ~6.4 GiB of an 8 GiB card
    assert tm.allocator_fraction(8.0, 8.0) == pytest.approx(0.92)                # never the whole card
