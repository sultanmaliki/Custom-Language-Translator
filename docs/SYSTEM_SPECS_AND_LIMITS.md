# System specs, limits and capacity

What this machine can and can't do for the Nawayathi translator, measured on **2026-09-26** with
[benchmark.py](../benchmark.py) and the real training script. Re-measure after a driver, hardware or
major library change (see [How these numbers were measured](#how-these-numbers-were-measured)).
Numbers marked *(estimate)* were worked out by arithmetic, not run. Timings vary by roughly ±30%
from run to run on a laptop, so treat them as ranges.

## TL;DR

- **Compute is not your bottleneck; data is.** Fine-tuning runs at roughly **40-90 sentence examples
  per second** depending on sentence length (each pair is two examples, one per direction). A
  5,000-pair corpus trains for 20 epochs in about **40-80 minutes**. You will run out of Nawayathi
  sentences long before you run out of GPU.
- **The one hard limit is the 8 GiB GPU** (about 6.9 GiB free). It is the reason for two settings you
  should know about: keep `batch size × longest sentence` under about **900 tokens**, and let
  `train_model.py` cap PyTorch's memory use. Past the limit, Windows doesn't raise an error; it
  silently borrows system RAM and training slows down (see [below](#windows-silently-borrows-system-ram)).
  `train_model.py` now warns you and guards against this.
- The current model (NLLB-600M + LoRA) fits comfortably. A 1.3B model *might* fit at small batches
  *(estimate)*. A 3.3B model won't.

## Hardware and software

| | |
| --- | --- |
| Machine | HP OMEN Gaming Laptop 16-ae0xxx |
| CPU | Intel Core i7-14650HX, 16 cores / 24 threads |
| RAM | 15.7 GiB (only ~6 GiB was free with normal desktop apps open) |
| GPU | NVIDIA GeForce RTX 4060 **Laptop**, 8 GiB VRAM (~6.9 GiB free; Windows itself uses ~1.1 GiB), compute capability 8.9, 24 SMs, max power limit 120 W |
| Disk | C: 486 GiB free |
| OS | Windows 11 Home (build 26200), power plan "Balanced", tested on AC power |
| Driver / CUDA | NVIDIA driver 617.14; PyTorch's CUDA runtime 12.8, cuDNN 9.19 |
| Python stack | Python 3.14.7, torch 2.11.0+cu128, transformers 5.17.0, peft 0.21.0, gradio 6.28.0 |
| bf16 | Supported (used for the frozen base model and autocast) |

Disk use: project `.venv` 4.6 GiB, Hugging Face cache 4.6 GiB. The NLLB-600M model is only 2.5 GB, so
the cache is nearly double that, most likely because Windows Developer Mode is off and symlinks
aren't available (Hugging Face warns about exactly this); turning Developer Mode on should avoid it
for future downloads. Each trained adapter is only about 35 MiB.

## GPU memory: the limit that matters

Training memory is dominated by NLLB's 256,000-word output layer, so it grows with the number of
**target tokens per batch** (`batch size × longest sentence in the batch`), not with model size.
Measured with `python benchmark.py train` (LoRA, bf16, the real training code, synthetic batches of
one fixed length; "peak" is GiB in use):

| sentence length | batch 8 | batch 16 | batch 32 | batch 64 |
| --- | --- | --- | --- | --- |
| 32 tokens | 50 ex/s, 2.4 | 89 ex/s, 3.5 | 132 ex/s, 5.8 | 23 ex/s, 10.2 ⚠️ |
| 64 tokens | 47 ex/s, 3.5 | 68 ex/s, 5.8 | 12 ex/s, 10.2 ⚠️ | not run |
| 96 tokens | 40 ex/s, 4.6 | 15 ex/s, 8.0 ⚠️ | not run | not run |

⚠️ = too much memory: PyTorch reserved more than the GPU has, so it spilled into system RAM. (The
benchmark stops a column at the first size that spills.)

**Rule of thumb**, fitted to these runs (within 0.1 GiB on every point):

> peak GPU memory ≈ 1.3 GiB + 4.4 MB × (batch size × longest target length)

`train_model.py` uses this to check your settings before it starts, and prints a warning with a
suggested `--batch-size` and `--accum` if the estimate goes above ~70% of the free memory (about
5.2 GiB, or roughly 900 tokens per batch). The project defaults (`--batch-size 8`, `--max-len 96` =
768 tokens) are safe. `python benchmark.py train` prints the estimate next to each measured peak so
you can check it on any GPU.

## Windows silently borrows system RAM

The card has 8 GiB, yet some rows above peak at 10 GiB. On Windows the NVIDIA driver lets CUDA
overflow into shared system memory instead of raising an out-of-memory error, so nothing fails; it
just gets slow. Two things worth knowing:

**1. The slowdown is gradual, and it starts when PyTorch's *reserved* memory passes the GPU's free
memory (~6.9 GiB), not its in-use memory.** Reserved runs about 0.6 GiB above in-use. Batch 16 at
increasing sentence lengths (`benchmark.py train --lengths 48 56 64 72 80 --batches 16`):

| length | in use | reserved | speed |
| --- | --- | --- | --- |
| 48 | 4.65 GiB | 5.11 GiB | 74 ex/s |
| 56 | 5.22 GiB | 5.73 GiB | 70 ex/s |
| 64 | 5.75 GiB | 6.30 GiB | 66 ex/s |
| 72 | 6.36 GiB | 7.02 GiB (just over) | 59 ex/s ⚠️ |
| 80 | 6.87 GiB | 7.59 GiB | 32 ex/s ⚠️ (half speed) |
| 96 | 8.0 GiB | 8.9 GiB | 15 ex/s ⚠️ (5x slower) |

**2. Real training is worse than fixed-size batches, because every batch has a different padded
length.** PyTorch's allocator caches memory by block size, and since it never sees an out-of-memory
error it never learns to free that cache, so the cache balloons far past the GPU even though the
memory actually in use is small. With the project's defaults (batch 8, lengths varying from 8 to 96
tokens, `benchmark.py varlen`), only 4.6 GiB was in use but **12.7 GiB was reserved, and it ran at
14 ex/s.**

**The fix, built into `train_model.py`:** pad every batch to a multiple of 8, so batches share a few
shapes, and cap PyTorch's memory at 92% of the VRAM that is really free, so it frees its cache when
needed. The same test then ran at **43 ex/s with 6.4 GiB reserved** (3x faster). Neither fix works
alone: padding alone still spilled, and the cap alone stopped the spilling but only reached 18 ex/s.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` has no effect on Windows.

**On the real script** (`train_model.py` on a 640-pair corpus of joined toy sentences, longest target
85 tokens, 5 epochs), the fix made training about 20% faster and cut reserved memory in half:

| | total time | reserved by PyTorch |
| --- | --- | --- |
| before | 252 s | 10.6 GiB (well past the 8 GiB card) |
| after (current) | 200 s | 5.1 GiB |

On a corpus with shorter sentences (longest target 48 tokens) there was no difference in speed
(about 36 s per epoch either way); reserved memory still dropped from 5.1 to 4.1 GiB. So the fix
matters most when you have long, varied sentences, and it stops training from eating your system RAM
(which is often only ~6 GiB free). The 3x figure above is the worst case, not the typical gain.

## Training capacity (NLLB-600M + LoRA, 8.65M trainable parameters)

Speed depends mostly on sentence length. From the runs above: **~90 ex/s at 32 tokens, ~50-70 at 64,
~40 at 96.** A 150-second sustained run (batch 16 × 48 tokens) held **~80 ex/s** at 73 °C, 2,595 MHz,
~72 W, with no throttling. Real Nawayathi sentences are likely short (the starter set averages about
5 words), so the left column below is the likely one:

| corpus (sentence pairs) | short sentences (~90 ex/s) | long sentences (~40 ex/s) |
| --- | --- | --- |
| 1,000 | 22 s per epoch, 7 min for 20 | 50 s per epoch, 17 min for 20 |
| 5,000 | 2 min per epoch, 37 min for 20 | 4 min per epoch, 83 min for 20 |
| 20,000 | 7 min per epoch, 2.5 h for 20 | 17 min per epoch, 5.6 h for 20 |
| 100,000 | 37 min per epoch, 6 h for 10 | 83 min per epoch, 14 h for 10 |

Early stopping usually ends a run before 20 epochs. In practice you can do **dozens of experiments a
day** at 1,000 pairs or **~10 a day** at 5,000 pairs, enough to compare language tags
(`NWY_TAG` in `config.py`), learning rates and LoRA sizes.

## Inference capacity (beam search 5, `python benchmark.py infer`)

Generating exactly 30 tokens per sentence, so it doesn't depend on what the model says. Merging the
trained adapter doesn't change the cost.

| | GPU | CPU only (`--cpu`) |
| --- | --- | --- |
| Model load | 6.6 s | 3.6 s |
| Memory | 1.15 GiB VRAM | 1.8 GiB RAM (3.2 GiB with the adapter loaded through `inference.py`) |
| One sentence | 0.46 s (15 ms per token) | 2.5 s (85 ms per token) |
| Batch of 16 | 0.66 s, 24 sentences/s | 12 s, 1.3 sentences/s |
| Batch of 64 | 1.5 s, 41 sentences/s (3.2 GiB) | not run |

A short sentence (about 8 output tokens) took 126 ms on the GPU and 540 ms on the CPU through
`inference.py`. The web app handles requests one at a time on a single loaded model, which is fine for
a handful of users. CPU-only translation is a usable fallback while the GPU is training; force it with
`CUDA_VISIBLE_DEVICES=-1` (an *empty* value doesn't hide the GPU from PyTorch).

## What fits and what doesn't

| Option | Verdict |
| --- | --- |
| NLLB-200-distilled-**600M** + LoRA (current) | Fits with room to spare. Measured. |
| NLLB-200-distilled-**1.3B** + LoRA | Weights ~2.6 GiB in bf16, so training would need batch × length ≲ 300-400 and run ~2-2.5x slower *(estimate, untested; ~5.5 GB download)*. Worth trying only once the 600M model is clearly limited by model size rather than data. |
| NLLB-200 **3.3B** | Weights alone are ~6.2 GiB in bf16, leaving no room for training or generation in ~6.9 GiB free. Would need 4-bit quantisation (QLoRA), which is untested on Windows + Python 3.14. Not recommended. |
| Full fine-tuning of the 600M model (all weights) | Weights, gradients and Adam state need ~7 GiB before any activations *(estimate)*. Doesn't fit, and LoRA is the better choice for a tiny corpus anyway. |
| Training from scratch | Hardware could run it, but it needs orders of magnitude more data than exists for Nawayathi. |

## Limitations and gotchas

1. **Stay plugged in.** All measurements were on AC power. On battery, expect the GPU to be
   power-limited *(not measured)*. `benchmark.py` warns if it detects battery. Keep vents clear for
   long runs; the GPU peaked at 74 °C here.
2. **Sleep or restart kills a run.** `train_model.py` has no resume feature. The best adapter so far is
   kept on disk, but optimizer state is lost. Disable sleep during long runs, and mind Windows Update restarts.
3. **Don't train and serve at once.** Training can use ~5 GiB and the app ~1.2 GiB. It might fit, but
   it slows both and risks the system-RAM overflow described above. Use CPU-only serving if you need both.
4. **RAM is modest.** 15.7 GiB total, often ~6 GiB free with your usual apps open. A CPU-only run of
   the model held up to 3.2 GiB, and the NLLB checkpoint is 2.5 GB on disk; close heavy apps if
   loading is slow or fails.
5. **Python 3.14 is new.** Everything used so far works, but libraries that ship compiled GPU code
   (bitsandbytes, flash-attn, Triton-based tools) may lag behind. `torch.compile` and 4-bit/8-bit
   quantisation are untested here.
6. **The 256k-word vocabulary is the real cost driver.** Almost all of the memory and much of the time
   per step comes from the output layer over words Nawayathi never uses. Trimming the vocabulary to the
   words actually seen would likely allow much larger batches and faster training *(estimate,
   untested; needs engineering)*. The best lever if this machine ever feels too small.
7. **Benchmark artifacts to avoid.** Comparing configurations in one process without clearing
   PyTorch's cache (`torch.cuda.empty_cache()`) makes later cells look slower than they are; an early
   version of my grid did this and showed spills that weren't real. `benchmark.py` now clears it
   between cells. Run each `varlen` configuration in its own process.
8. **Licence.** NLLB's weights are CC-BY-NC 4.0 (non-commercial); adapters trained from them inherit that.

## How these numbers were measured

Everything comes from `benchmark.py` in the repo root (plug the laptop in first), except the
real-script comparison, which is `train_model.py` itself:

```
python benchmark.py specs                                   # hardware and software summary
python benchmark.py train                                   # batch size x length grid: speed, memory, estimate
python benchmark.py varlen --batch-size 8 --max-length 96   # a different length every batch, like real training
python benchmark.py varlen --batch-size 8 --max-length 96 --memory-fraction 0.8 --pad-multiple 8   # with the fixes
python benchmark.py sustained --seconds 150                 # thermal / power throttling
python benchmark.py infer                                   # translation speed (add --cpu for CPU only)
```

Training benchmarks use synthetic batches through the project's own `make_collate` / `batch_loss` on
the LoRA-wrapped model: 2 warm-up steps, then 5 timed steps per cell. Inference uses fixed-length
generation to isolate per-token cost, on the base model (no adapter needed). The real-script
comparison ran `train_model.py` on a generated corpus of 1-11 toy sentences joined per pair, once with
`PAD_MULTIPLE` set to 1 and the memory cap disabled, once as shipped.
