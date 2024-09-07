from keras.layers import Add, RepeatVector, Concatenate

def attention_model(inputs, features):
    attention = Dense(1, activation='tanh')(inputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(256)(attention)
    attention = Permute([2, 1])(attention)

    merged = Concatenate()([features, attention])
    return merged

# Apply attention to the LSTM model
def define_model_with_attention(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    features = Dense(256, activation='relu')(inputs1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, return_sequences=True)(se2)

    attention = attention_model(se3, features)
    
    decoder2 = LSTM(256)(attention)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model
