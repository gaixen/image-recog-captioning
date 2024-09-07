pip install tensorflow keras numpy pillow
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, RepeatVector, Concatenate, TimeDistributed, Activation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# Load InceptionV3 model pre-trained on ImageNet data
inception_model = InceptionV3(weights='imagenet')
model_new = Model(inception_model.input, inception_model.layers[-2].output)

# Function to extract features from an image
def extract_features(image_path):
    img = load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_new.predict(x, verbose=0)
    return features

# Define the attention mechanism
def attention_layer(inputs, features):
    attention = Dense(256, activation='tanh')(inputs)
    attention = Dense(1)(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(256)(attention)
    attention = tf.transpose(attention, [0, 2, 1])
    context = tf.matmul(attention, features)
    return context

# Define the Image Captioning Model with Attention
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256, return_sequences=True)(se2)
    
    context = attention_layer(se3, fe2)
    decoder = LSTM(256)(context)
    outputs = Dense(vocab_size, activation='softmax')(decoder)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Assuming you have a pre-trained model and tokenizer
# Load your pre-trained model (replace with actual file paths)
vocab_size = 10000  # Example vocabulary size
max_length = 34  # Example maximum caption length
model = define_model(vocab_size, max_length)

# Dummy loading, replace with actual trained model and tokenizer
# model.load_weights('model_weights.h5')
# tokenizer = load_your_tokenizer()  # Load your trained tokenizer

# Function to generate caption
def generate_caption(model, tokenizer, image_features, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Upload an image and generate caption
def upload_and_caption(image_path):
    image_features = extract_features(image_path)
    caption = generate_caption(model, tokenizer, image_features, max_length)
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    return caption

# Example usage
image_path = 'example.jpg'  # Replace with the path to your image
caption = upload_and_caption(image_path)
print(f"Generated Caption: {caption}")

   
