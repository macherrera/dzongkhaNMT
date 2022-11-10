# Imports
from urllib.request import ProxyBasicAuthHandler
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Bidirectional, GRU, LSTM, Dense, RepeatVector, Input, Embedding, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.losses import sparse_categorical_crossentropy
import tensorflow.keras.callbacks
import string
import os
from datetime import datetime
import pickle
import numpy as np
from functools import lru_cache
import json
from tensorflow.math import argmax
import re
import sys
#################################
# Requires files:
# "PROBS_json"
# "eng_tokenizer.pickle"
# "dzo_tokenizer.pickle",
# "model.h5"
# to be stored in dir "Data"
#################################

def main(s):
    # Define constants
    BASE_DIR = "Data"
    PROBS_json = "PROBS_json"
    PATH_TO_MODEL = os.path.join(BASE_DIR, "model.h5")
    TSHEG = r"་"
    max_dzo_sentence_size = 48  # based on available corpus (needs to be updated if training set changes)
    eng_tokenizer_path = os.path.join(BASE_DIR, 'eng_tokenizer.pickle')
    dzo_tokenizer_path = os.path.join(BASE_DIR, 'dzo_tokenizer.pickle')

    # Load model
    model = load_model(PATH_TO_MODEL)
    # Load tokenizers
    with open(os.path.join(eng_tokenizer_path), 'rb') as handle:
        eng_tokenizer = pickle.load(handle)
    with open(os.path.join(dzo_tokenizer_path), 'rb') as handle:
        dzo_tokenizer = pickle.load(handle)

    # Preprocessing: Step 1
    # s = sys.argv[1]
    print(f"Translating sentence: {s} from Dzongkha to English")
    print("length", len(s))
    s = s.replace(" ", "")  # get rid of spaces because they are not significant


    # Obtain segmentation data
    def get_dzo_frequencies():
        """Opens precomputed frequencies for Dzongkha words as JSON.
    Returns the probabilities as dict, the len of the longest word and the sum of all probs"""
        with open(os.path.join(BASE_DIR, PROBS_json), "r") as f:
            probs = json.load(f)
        max_len = max([len(w) for w in probs.keys()])
        n = sum(probs.values())
        return probs, max_len, n
    PROBS, MAX_LEN, N = get_dzo_frequencies()

    # Define Dzongkha segmentation function
    @lru_cache(maxsize=None, typed=True)
    def segment(text, probs=PROBS, max_len=MAX_LEN, n=N):
        """Given a text as a non segmented string returns a segmented space separated string. Implements Novig's algorithm
    """
        text = text.lower()
        candidates = []
        if not text:
            return []
        for first, other in [(text[:i + 1], text[i + 1:]) for i in range(min(len(text), max_len))]:
            candidates.append([first] + segment(other))
        return max(candidates,
                   key=lambda x: np.prod(np.array([probs.get(word, 10. / (n * 10 ** len(word))) for word in x])))

    # Preprocessing step 2
    to_replace = " " + TSHEG + " "  # get rid of TSHEG between words
    input_sentence = " ".join(segment(s))  # Segment
    input_sentence = input_sentence.replace(to_replace, " ")
    input_sentence = ' '.join(input_sentence.split())  # Join together

    # Vectorize and pad input sentence
    input_sentence_vectorized = dzo_tokenizer.texts_to_sequences([input_sentence])
    input_sentence_padded = pad_sequences(
        input_sentence_vectorized, max_dzo_sentence_size, padding="post")

    # Obtain prediction
    predicted_sentence_vector = model.predict(input_sentence_padded)[0]
    predicted_sentence_candidates = argmax(
        predicted_sentence_vector, axis=1).numpy()

    # Function to turn predicted vector back into sentence
    def vector_to_sentence(tokenizer, vector):
        '''
    Given a vector and a Keras Tokenizer object instance, 
    uses the tokenizer's word_index to output a natural language sentence.
    '''
        reverse_dict = {v: k for k, v in tokenizer.word_index.items()}
        return (" ").join([reverse_dict[token] for token in vector if token])

    # Convert predicted vector to sentence
    out_sentence = vector_to_sentence(eng_tokenizer, predicted_sentence_candidates)

    return out_sentence


if __name__ == "__main__":
    print(main())