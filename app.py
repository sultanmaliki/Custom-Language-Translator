"""Web app: translate in either direction, and grow the corpus.

    python app.py

The Contribute tab works even before any model is trained, so you can start
collecting sentence pairs on day one.
"""
import threading

import gradio as gr

from data_preprocessing import append_pair, corpus_size
from inference import ModelNotTrainedError, Translator, split_sentences

DIRECTIONS = [("English → Nawayathi", "en2nwy"), ("Nawayathi → English", "nwy2en")]

_translator: Translator | None = None
_load_lock = threading.Lock()


def get_translator() -> Translator:
    """Load the model on first use. A failed load isn't cached, so the app picks
    up a model you finish training while it is running."""
    global _translator
    with _load_lock:
        if _translator is None:
            _translator = Translator()
        return _translator


def translate(text: str, direction: str):
    if not text.strip():
        return "", ""
    try:
        translator = get_translator()
    except ModelNotTrainedError as e:
        raise gr.Error(str(e))
    best = translator.translate(text, direction)
    alternatives = ""
    if len(split_sentences(text)) == 1 and "\n" not in text.strip():   # candidates only make sense per sentence
        others = [a for a in translator.alternatives(text, direction) if a != best]
        alternatives = "\n".join(others)
    return best, alternatives


def _pairs_label() -> str:
    n = corpus_size()
    return f"{n} pair{'' if n == 1 else 's'}"


def _save(english: str, nawayathi: str, source: str) -> str:
    try:
        status = append_pair(english, nawayathi, source)
    except ValueError as e:
        return f"⚠️ {e}"
    if status == "duplicate":
        return f"That pair is already in the corpus ({_pairs_label()})."
    return f"✅ Added. The corpus now has {_pairs_label()}."


def submit_correction(source_text: str, correction: str, direction: str) -> str:
    english, nawayathi = (source_text, correction) if direction == "en2nwy" else (correction, source_text)
    return _save(english, nawayathi, "correction")


def submit_pair(english: str, nawayathi: str, contributor: str):
    message = _save(english, nawayathi, contributor)
    added = message.startswith("✅")
    return message, ("" if added else english), ("" if added else nawayathi)


with gr.Blocks(title="Nawayathi Translator") as demo:
    gr.Markdown(
        "# Nawayathi ⇄ English translator\n"
        "Nawayathi is written here in Roman letters. It is a very low-resource language, so treat "
        "translations as suggestions, and please correct the ones that are wrong."
    )
    with gr.Tab("Translate"):
        direction = gr.Radio(DIRECTIONS, value="en2nwy", label="Direction")
        source = gr.Textbox(lines=3, label="Text to translate")
        translate_btn = gr.Button("Translate", variant="primary")
        output = gr.Textbox(lines=3, label="Translation", interactive=False)
        alternatives = gr.Textbox(lines=3, label="Other candidates", interactive=False)
        with gr.Accordion("Wrong translation? Give the right one", open=False):
            correction = gr.Textbox(lines=2, label="Correct translation")
            correct_btn = gr.Button("Submit correction")
            correct_status = gr.Markdown()
        translate_btn.click(translate, [source, direction], [output, alternatives])
        source.submit(translate, [source, direction], [output, alternatives])
        correct_btn.click(submit_correction, [source, correction, direction], correct_status)

    with gr.Tab("Contribute"):
        gr.Markdown(
            "Add a sentence pair to the training corpus (`data/pairs.tsv`). Write Nawayathi in Roman "
            "letters, and use full, natural sentences. Several spellings of the same sentence are welcome."
        )
        c_english = gr.Textbox(lines=2, label="English")
        c_nawayathi = gr.Textbox(lines=2, label="Nawayathi (Roman letters)")
        c_name = gr.Textbox(label="Your name or where this came from (optional)")
        c_btn = gr.Button("Add to corpus", variant="primary")
        c_status = gr.Markdown()
        c_btn.click(submit_pair, [c_english, c_nawayathi, c_name], [c_status, c_english, c_nawayathi])

if __name__ == "__main__":
    print(f"Corpus size: {_pairs_label()}")
    demo.launch()
