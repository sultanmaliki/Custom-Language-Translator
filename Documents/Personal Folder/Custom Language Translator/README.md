# Custom Language Translator

A customizable neural machine translation (NMT) pipeline for creating and training language translators.  
This project includes tools for dataset preprocessing, model training with attention-based Seq2Seq architecture, inference, and a web interface using **Gradio**.

---

## 🚀 Features

- Flexible dataset format support (TSV)  
- Encoder-decoder model with **attention** implemented in **Keras/TensorFlow**  
- Preprocessing scripts for tokenization and data preparation  
- Training script with **EarlyStopping** and checkpoint saving  
- Inference module with **beam search decoding**  
- Interactive translation interface via **Gradio**  

---

## 📂 File Structure

- `dataset.tsv` — bilingual dataset file  
- `data_preprocessing.py` — script to preprocess and tokenize dataset  
- `model_definition.py` — neural network architecture definition  
- `train_model.py` — training pipeline  
- `inference.py` — sentence translation using trained model  
- `app.py` — Gradio app to run translations via web UI  
- Model weight files: `.h5`, `.keras`  
- Tokenizer files: `.pkl`  

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/sultanmaliki/Custom-Language-Translator.git
   cd Custom-Language-Translator
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Preparation

1. Prepare your dataset in **TSV format** with two columns (source sentence and target sentence).  
   Example: `translation_dataset.tsv`

   ```tsv
   hello   bonjour
   how are you?   comment ça va ?
   ```

2. Run preprocessing

   ```bash
   python data_preprocessing.py
   ```

---

## 🏋️ Training

Train the model using:

```bash
python train_model.py
```

---

## 🔎 Inference and Interface

1. **Command-line inference**

   ```bash
   python inference.py
   ```

2. **Launch Gradio Web App**

   ```bash
   python app.py
   ```

---

## 💡 Usage Example

Translate a sentence via CLI:

```bash
python inference.py
```

Or use the **Gradio web interface** launched via:

```bash
python app.py
```

---

## 🤝 Contribution

Contributions and feature requests are welcome!  
Please **fork the repository** and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For help or questions, please contact:

- 📩 Email: [connect@syedmohammedsultan.online](mailto:connect@syedmohammedsultan.online)  
- 📷 Instagram: [@sm.sultan.maliki](https://instagram.com/sm.sultan.maliki)  
- 🌐 Website: [syedmohammedsultan.online](https://syedmohammedsultan.online)  
