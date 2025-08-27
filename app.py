import gradio as gr
import inference  # Your inference.py module with translate() function

def translate_interface(text):
    return inference.translate(text)

demo = gr.Interface(
    fn=translate_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter English sentence here...", label="English Input"),
    outputs=gr.Textbox(label="Translated Output"),
    title="Neural Machine Translator",
    description="Enter English to get the attention-based LSTM NMT translation.",
    theme="soft",
    flagging_mode="never"
)

if __name__ == "__main__":
    print("Launching Gradio UI...")
    demo.launch()
