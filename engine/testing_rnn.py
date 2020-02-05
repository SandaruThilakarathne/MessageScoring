import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
vocab_size = 5000
max_length = 200
oov_tok = '<OOV>'


def get_predictions(txt):
    # txt = ["A WeWork shareholder has taken the company to court over the near-$1.7bn (£1.3bn) leaving package approved "
    #        "for ousted co-founder Adam Neumann."]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

    tokenizer.fit_on_texts(txt)
    # word_index = tokenizer.word_index

    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=max_length)
    model = tf.keras.models.load_model(r'/home/villvay/PycharmProjects/MessageScoring/engine/my_model.h5')
    # print(model.summary())
    pred = model.predict(padded)
    # print(pred.tolist()[0])
    pred_list = [float(format(pred_data, '.8f')) * 100 for pred_data in pred.tolist()[0]]
    print(max(pred_list))
    # exit()
    labels = ['sport', 'business', 'entertainment', 'politics', 'tech']
    # print(pred)
    print(labels[pred_list.index(max(pred_list))])

    return max(pred_list), labels[pred_list.index(max(pred_list))], pred_list, labels
