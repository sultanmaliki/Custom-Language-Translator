import numpy as np
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_definition import build_model_with_attention

# 1. Load preprocessed data
print("Loading preprocessed data...")
encoder_input_data = np.load('encoder_input_data.npy')
decoder_input_data = np.load('decoder_input_data.npy')
decoder_output_data_onehot = np.load('decoder_output_data_onehot.npy')

print(f"Encoder input shape: {encoder_input_data.shape}")
print(f"Decoder input shape: {decoder_input_data.shape}")
print(f"Decoder output one-hot shape: {decoder_output_data_onehot.shape}")

# 2. Load config
print("Loading configuration...")
with open('data_config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

max_encoder_seq_length = config['max_encoder_seq_length']
max_decoder_seq_length = config['max_decoder_seq_length']
num_encoder_tokens = config['num_encoder_tokens']
num_decoder_tokens = config['num_decoder_tokens']

print(f"Config loaded: max_encoder_seq_length={max_encoder_seq_length}, max_decoder_seq_length={max_decoder_seq_length}")
print(f"Vocabulary sizes: num_encoder_tokens={num_encoder_tokens}, num_decoder_tokens={num_decoder_tokens}")

# 3. Build model
print("Building the model...")
model = build_model_with_attention(
    max_encoder_seq_length=max_encoder_seq_length,
    max_decoder_seq_length=max_decoder_seq_length,
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens,
    embedding_dim=512,
    latent_dim=1024,
    dropout_rate=0.3
)

model.summary()

# 4. Callbacks: early stopping and save best model checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_translation_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# 5. Train model
print("Training the model...")
batch_size = 32
epochs = 100

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data_onehot,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# 6. Save final model
model.save('translation_model.keras')
print("Model trained and saved as 'translation_model.keras'")
