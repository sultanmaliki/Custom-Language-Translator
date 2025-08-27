import pandas as pd
import numpy as np

# Try different Tokenizer imports for compatibility.
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# === 1. Load Dataset (TSV) ===
data = pd.read_csv('translation_dataset.tsv', sep='\t')  # note sep='\t' for TSV
input_texts = data['source_sentence'].astype(str).tolist()
target_texts = data['target_sentence'].astype(str).tolist()

# === 2. Prepare target sequences with start/end tokens ===
target_texts_input = ['<start> ' + text for text in target_texts]
target_texts_output = [text + ' <end>' for text in target_texts]

# === 3. Tokenization ===
# Source tokenizer (English)
input_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
max_encoder_seq_length = max(len(seq) for seq in input_sequences)
num_encoder_tokens = len(input_tokenizer.word_index) + 1

# Target tokenizer (your language)
target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(target_texts_input + target_texts_output)
target_sequences_input = target_tokenizer.texts_to_sequences(target_texts_input)
target_sequences_output = target_tokenizer.texts_to_sequences(target_texts_output)
max_decoder_seq_length = max(len(seq) for seq in target_sequences_input)
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# === 4. Padding ===
encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(target_sequences_input, maxlen=max_decoder_seq_length, padding='post')
decoder_output_data = pad_sequences(target_sequences_output, maxlen=max_decoder_seq_length, padding='post')

# === 5. One-hot Encoding for decoder output ===
decoder_output_data_onehot = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32'
)
for i, seqs in enumerate(decoder_output_data):
    for t, word_id in enumerate(seqs):
        if word_id > 0:
            decoder_output_data_onehot[i, t, word_id] = 1.0

# === 6. Save preprocessed arrays and tokenizers ===
np.save('encoder_input_data.npy', encoder_input_data)
np.save('decoder_input_data.npy', decoder_input_data)
np.save('decoder_output_data_onehot.npy', decoder_output_data_onehot)

import pickle, json
with open('input_tokenizer.pkl', 'wb') as f:
    pickle.dump(input_tokenizer, f)
with open('target_tokenizer.pkl', 'wb') as f:
    pickle.dump(target_tokenizer, f)
with open('data_config.json', 'w') as f:
    json.dump({
        'max_encoder_seq_length': max_encoder_seq_length,
        'max_decoder_seq_length': max_decoder_seq_length,
        'num_encoder_tokens': num_encoder_tokens,
        'num_decoder_tokens': num_decoder_tokens,
    }, f)

print("✅ Data preprocessing complete. Arrays and tokenizers saved.")
