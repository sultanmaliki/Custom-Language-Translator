import numpy as np
import json
import pickle
import tensorflow as tf
from keras.models import Model
from keras.layers import Input

# Model parameters - ensure these match training
latent_dim = 1024
embedding_dim = 512

# Load trained model
model = tf.keras.models.load_model('translation_model.keras')

# Load tokenizers and config
with open('input_tokenizer.pkl', 'rb') as f:
    input_tokenizer = pickle.load(f)
with open('target_tokenizer.pkl', 'rb') as f:
    target_tokenizer = pickle.load(f)
with open('data_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

max_encoder_seq_length = config['max_encoder_seq_length']
max_decoder_seq_length = config['max_decoder_seq_length']

# Rebuild encoder for inference - outputs encoder sequence + states
encoder_inputs = model.input[0]
encoder_layer = model.get_layer('encoder_lstm2')
encoder_embedding_layer = model.get_layer('encoder_embedding')
encoder_embedded = encoder_embedding_layer(encoder_inputs)
encoder_outputs, state_h_enc, state_c_enc = encoder_layer(encoder_embedded)
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h_enc, state_c_enc])

# Rebuild decoder for inference with attention
decoder_inputs = model.input[1]
decoder_state_h_input = Input(shape=(latent_dim,), name='input_state_h')
decoder_state_c_input = Input(shape=(latent_dim,), name='input_state_c')
encoder_outputs_input = Input(shape=(max_encoder_seq_length, latent_dim), name='encoder_outputs_input')

decoder_states_inputs = [decoder_state_h_input, decoder_state_c_input]

decoder_embedding_layer = model.get_layer('decoder_embedding')
decoder_lstm1_layer = model.get_layer('decoder_lstm1')
decoder_lstm2_layer = model.get_layer('decoder_lstm2')
attention_layer = model.get_layer('attention_layer')
concat_layer = model.get_layer('concat_layer')
dense_layer = model.get_layer('time_distributed_dense')

dec_emb = decoder_embedding_layer(decoder_inputs)
dec_lstm1_out, state_h1, state_c1 = decoder_lstm1_layer(dec_emb, initial_state=decoder_states_inputs)
dec_lstm2_out, state_h2, state_c2 = decoder_lstm2_layer(dec_lstm1_out)

attended_context = attention_layer([dec_lstm2_out, encoder_outputs_input])
decoder_concat = concat_layer([dec_lstm2_out, attended_context])
decoder_outputs_final = dense_layer(decoder_concat)

decoder_model = Model(
    inputs=[decoder_inputs, encoder_outputs_input] + decoder_states_inputs,
    outputs=[decoder_outputs_final, state_h1, state_c1, state_h2, state_c2]
)

def decode_sequence(input_seq):
    # Encode input to get encoder outputs and states
    encoder_outs, state_h, state_c = encoder_model.predict(input_seq)
    states_value = [state_h, state_c]

    # Prepare empty target sequence with start token
    target_seq = np.zeros((1, 1), dtype='int32')
    start_token_index = target_tokenizer.word_index.get('<start>')
    if start_token_index is None:
        raise ValueError("'<start>' token not found in target tokenizer.")
    target_seq[0, 0] = start_token_index

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, encoder_outs] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_decoder_seq_length:
            stop_condition = True
        elif sampled_word == '':
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            target_seq = np.zeros((1, 1), dtype='int32')
            target_seq[0, 0] = sampled_token_index
            states_value = [h2, c2]

    return decoded_sentence.strip()

def translate(sentence):
    seq = input_tokenizer.texts_to_sequences([sentence])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_encoder_seq_length, padding='post')
    return decode_sequence(seq)

if __name__ == '__main__':
    print("Enter English sentences to translate (empty line to quit)")
    while True:
        text = input("Input: ").strip()
        if not text:
            print("Exiting...")
            break
        try:
            translation = translate(text)
            print(f"Translation: {translation}")
        except Exception as e:
            print(f"Error: {e}")
