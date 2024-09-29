import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
import re

# Define parameters
batch_size = 64
epochs = 100
latent_dim = 256

# Path to the CSV file on disk
csv_file_path = 'C:/PROJECT/Conversation.csv'

# Read CSV file
df = pd.read_csv('C:/PROJECT/Conversation.csv')

# Assuming the columns are named differently, adjust accordingly
input_column_name = 'question'  # Adjust this if the column name is different
target_column_name = 'answer'    # Adjust this if the column name is different

# Preprocess the input and target texts
df[input_column_name] = df[input_column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))
df[target_column_name] = df[target_column_name].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

# Convert dataframe columns to lists
input_texts = df[input_column_name].tolist()
target_texts = df[target_column_name].tolist()

# Vectorize the data
input_tokenizer = tf.keras.preprocessing.text.Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_maxlen = max(len(seq) for seq in input_sequences)
input_vocab_size = len(input_tokenizer.word_index) + 1

target_tokenizer = tf.keras.preprocessing.text.Tokenizer()
target_tokenizer.fit_on_texts(target_texts)
target_texts = ['<start> ' + text + ' <end>' for text in target_texts]  # Add <start> and <end> tokens
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_maxlen = max(len(seq) for seq in target_sequences)
target_vocab_size = len(target_tokenizer.word_index) + 1

encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=input_maxlen, padding='post')
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=target_maxlen, padding='post')

decoder_target_data = np.zeros((len(target_sequences), target_maxlen, target_vocab_size), dtype='float32')
for i, seqs in enumerate(target_sequences):
    for j, seq in enumerate(seqs):
        decoder_target_data[i, j, seq] = 1.0

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_vocab_size, latent_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, latent_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
# Save model with .keras extension instead of .h5
model.save('chatbot_model.keras')

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.keras')

# Define a function to directly return the target text
def respond_to_input(input_text):
    # Convert user input to lowercase and remove symbols
    input_text = re.sub(r'[^a-zA-Z\s]', '', input_text.lower())
    
    # Check if the input exists in the dataset
    if input_text in input_texts:
        # Get the index of the input_text in input_texts
        input_index = input_texts.index(input_text)
        
        # Retrieve the corresponding target_text from target_texts
        target_text = target_texts[input_index]
        
        # Remove '<start>' and '<end>' tokens from target_text
        target_text = ' '.join(target_text.split()[1:-1])
        
        return target_text
    else:
        return "Sorry, I don't understand."

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    bot_response = respond_to_input(user_input)
    print("Bot:", bot_response)
