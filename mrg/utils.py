import os
import _pickle as pkl

import numpy as np
import tensorflow as tf

from reader import PAD_WORD, START_WORD, END_WORD
import pandas as pd
import tensorflow as tf
from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
from gensim.models import KeyedVectors
FLAGS = tf.flags.FLAGS


def log_info(log_file, msg):
    print(msg)
    log_file.write('{}\n'.format(msg))


def get_shape(tensor):
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims


def count_parameters(trained_vars):
    total_parameters = 0
    print('=' * 100)
    for variable in trained_vars:
        variable_parameters = 1
        for dim in variable.get_shape():
            variable_parameters *= dim.value
        print('{:70} {:20} params'.format(variable.name, variable_parameters))
        print('-' * 100)
        total_parameters += variable_parameters
    print('=' * 100)
    print("Total trainable parameters: %d" % total_parameters)
    print('=' * 100)


def load_glove(vocab_size, embedding_size):
    # return np.zeros((vocab_size, embedding_size))
    print('Loading pre-trained word embeddings')
    embedding_weights = {}
    f = open('glove.6B.{}d.txt'.format(embedding_size), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_weights[word] = coefs
    f.close()
    print('Total {} word vectors in Glove 6B {}d.'.format(
        len(embedding_weights), embedding_size))

    embedding_matrix = np.random.normal(0, 0.01, (vocab_size, embedding_size))

    vocab = pkl.load(open(os.path.join(FLAGS.data_dir, 'vocab.pkl'), 'rb'))
    oov_count = 0
    for word, i in vocab.items():
        embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
    print('Number of OOV words: %d' % oov_count)

    return embedding_matrix


def load_vocabulary(data_dir):
    vocab_file = os.path.join(data_dir, 'vocab.pkl')
    if not os.path.exists(vocab_file):
        raise FileNotFoundError('Vocabulary not found: %s' % vocab_file)

    print('Reading vocabulary: %s' % vocab_file)
    try:
        with open(vocab_file, 'rb') as f:
            return {idx: word for word, idx in pkl.load(f).items()}
    except IOError:
        pass


def decode_reviews(reviews, vocab):
    if reviews.ndim == 1:
        T = reviews.shape[0]
        N = 1
    else:
        N, T = reviews.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if reviews.ndim == 1:
                word = vocab[reviews[t]]
            else:
                word = vocab[reviews[i, t]]
            if word == END_WORD:
                break
            if word != START_WORD and word != PAD_WORD:
                words.append(word)
        decoded.append(words)
    return decoded


def predict_similarity(sentence1, sentence2, word2vecmodel):
    d = {'test_id': 0, 'question1': sentence1, 'question2': sentence2}
    test_df = pd.DataFrame(data=d, index=[0])

    for c, q in enumerate(['question1', 'question2']):
        test_df[q + '_n'] = test_df[q]

    embedding_dim = 300
    max_seq_length = 20

    test_df, embeddings = make_w2v_embeddings(
        test_df, word2vecmodel, embedding_dim=embedding_dim, empty_w2v=False)

    # Split to dicts and append zero padding.
    X_test = split_and_zero_padding(test_df, max_seq_length)

    # Make sure everything is ok
    assert X_test['left'].shape == X_test['right'].shape

    model = tf.keras.models.load_model(
        './data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
    model.summary()

    prediction = model.predict([X_test['left'], X_test['right']])
    return prediction
