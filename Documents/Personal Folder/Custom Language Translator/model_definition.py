from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Dropout, AdditiveAttention, Concatenate, TimeDistributed

def build_model_with_attention(
    max_encoder_seq_length,
    max_decoder_seq_length,
    num_encoder_tokens,
    num_decoder_tokens,
    embedding_dim=512,
    latent_dim=1024,
    dropout_rate=0.3
):
    # Encoder
    encoder_inputs = Input(shape=(max_encoder_seq_length,), name='encoder_inputs')
    encoder_embedding = Embedding(
        num_encoder_tokens,
        embedding_dim,
        mask_zero=False,
        name='encoder_embedding'
    )(encoder_inputs)
    encoder_emb_dropout = Dropout(dropout_rate)(encoder_embedding)

    encoder_lstm1 = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name='encoder_lstm1'
    )
    enc_out1, enc_h1, enc_c1 = encoder_lstm1(encoder_emb_dropout)

    encoder_lstm2 = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name='encoder_lstm2'
    )
    encoder_outputs, enc_h2, enc_c2 = encoder_lstm2(enc_out1)

    encoder_states = [enc_h2, enc_c2]

    # Decoder
    decoder_inputs = Input(shape=(max_decoder_seq_length,), name='decoder_inputs')
    decoder_embedding = Embedding(
        num_decoder_tokens,
        embedding_dim,
        mask_zero=False,
        name='decoder_embedding'
    )(decoder_inputs)
    decoder_emb_dropout = Dropout(dropout_rate)(decoder_embedding)

    decoder_lstm1 = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name='decoder_lstm1'
    )
    dec_out1, dec_h1, dec_c1 = decoder_lstm1(decoder_emb_dropout, initial_state=encoder_states)

    decoder_lstm2 = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        name='decoder_lstm2'
    )
    decoder_outputs, _, _ = decoder_lstm2(dec_out1)

    # Attention layer
    attention = AdditiveAttention(name='attention_layer')
    attended_context = attention([decoder_outputs, encoder_outputs])

    # Concatenate context and decoder outputs
    concat_layer = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attended_context])

    # Dense output layer
    dense = TimeDistributed(Dense(num_decoder_tokens, activation='softmax'), name='time_distributed_dense')
    decoder_outputs = dense(concat_layer)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
