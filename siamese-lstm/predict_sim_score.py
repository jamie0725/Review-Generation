import pandas as pd

import tensorflow as tf

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist
from gensim.models import KeyedVectors


def predict_similarity(sentence1, sentence2, word2vecmodel):
	d = {'test_id': 0, 'question1': sentence1, 'question2': sentence2}
	test_df = pd.DataFrame(data=d, index=[0])

	for c, q in enumerate(['question1', 'question2']):
	    test_df[q + '_n'] = test_df[q]

	embedding_dim = 300
	max_seq_length = 20
	
	test_df, embeddings = make_w2v_embeddings(test_df, word2vecmodel, embedding_dim=embedding_dim, empty_w2v=False)

	# Split to dicts and append zero padding.
	X_test = split_and_zero_padding(test_df, max_seq_length)

	# Make sure everything is ok
	assert X_test['left'].shape == X_test['right'].shape

	model = tf.keras.models.load_model('./data/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
	model.summary()

	prediction = model.predict([X_test['left'], X_test['right']])
	return prediction

if __name__ == "__main__":
	# Load word2vec
	print("Loading word2vec model(it may takes 2-3 mins) ...")

 	word2vecmodel = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True)

	prediction = predict_similarity('hey there', 'hey there', word2vecmodel)

	print(prediction)