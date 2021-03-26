import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import numpy as np
import pandas as pd


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import os
import re

path = os.getcwd()

movie_reviews = pd.read_csv(path+'/BERT/IMDB_Dataset.csv')
movie_reviews.isnull().values.any()


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

reviews = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    reviews.append(preprocess_text(sen))

print(movie_reviews.columns.values)

y = movie_reviews['sentiment']
y = np.array(list(map(lambda x:1 if x == 'positive' else 0, y)))


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

def tokenize_reviews(text_reviews):
    return tokenizer.convert_tokents_to_ids(tokenizer.tokenize(text_reviews))

tokenized_reviews = [tokenize_reviews(review) for review in reviews]
