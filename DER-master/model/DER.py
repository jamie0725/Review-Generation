import tensorflow as tf
import pickle
import time
import os

class DER():
    def __init__(self, args):
        tf.set_random_seed(0)

        self._null = 0
        self.ll = tf.constant(0.0)
        self.result = []
        self.parameters = args
        self.global_dimension = args.global_dimension  # 10
        self.word_dimension = args.word_dimension  # 10
        self.batch_size = args.batch_size  # 5
        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.root_path = args.root_path
        self.input_data_type = args.input_data_type
        self.output_path = args.output_path
        self.max_interaction_length = args.max_interaction_length
        self.max_sentence_length = args.max_sentence_length
        self.max_sentence_word_length = args.max_sentence_word_length
        self.time_bin_number = args.time_bin_number
        self.word_embedding_path = os.path.join(args.root_path, args.input_data_type, 'word_emb.pkl')
        self.logger = args.logger
        self.namespace = args.namespace
        self.emb = pickle.load(open(self.word_embedding_path, "rb"))
        self.word_vocab_size = len(self.emb)
        print self.word_vocab_size


        self.weight_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        self.constant_initializer = tf.constant_initializer(0.0)

        self.user_plh = tf.placeholder(tf.int32, shape=[None], name='user_plh')
        self.previous_items_plh = tf.placeholder(tf.int32, shape=[None, self.max_interaction_length], name='previous_items_plh')
        self.previous_times_plh = tf.placeholder(tf.int32, shape=[None, self.max_interaction_length], name='previous_times_plh')
        self.previous_reviews_plh = tf.placeholder(tf.float32, shape=[None, self.max_interaction_length, self.word_dimension], name='previous_reviews_plh')
        self.previous_ratings_plh = tf.placeholder(tf.int32, shape=[None, self.max_interaction_length], name='previous_ratings_plh')
        self.previous_lengths_plh = tf.placeholder(tf.int32, shape=[None], name='previous_lengths_plh')

        self.current_item_plh = tf.placeholder(tf.int32, shape=[None], name='current_item_plh')
        self.current_rating_plh = tf.placeholder(tf.float32, shape=[None], name='current_rating_plh')
        self.current_input_reviews_plh = tf.placeholder(tf.int32, shape=[None, self.max_sentence_length, self.max_sentence_word_length], name='current_input_reviews_plh')
        self.current_input_reviews_users_plh = tf.placeholder(tf.int32, shape=[None, self.max_sentence_length], name='current_input_reviews_users_plh')
        self.current_input_reviews_length_plh = tf.placeholder(tf.int32, shape=[None], name='current_input_length_plh')

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=tf.random_uniform_initializer(minval=-1, maxval=1), shape=[self.parameters.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.weight_initializer, shape=[self.parameters.item_number, self.global_dimension])
        self.rating_embedding_matrix = tf.get_variable('rating_embedding_matrix', initializer=self.weight_initializer, shape=[6, 5])
        self.emb_initializer = tf.placeholder(dtype=tf.float32, shape=[self.word_vocab_size, 64])
        self.word_embedding_matrix = tf.Variable(self.emb_initializer, trainable=False, collections=[], name='word_embedding')
        self.ll += self.parameters.reg * tf.nn.l2_loss(self.user_embedding_matrix) \
                   + self.parameters.reg * tf.nn.l2_loss(self.item_embedding_matrix)

        self.user_bias_vector = tf.get_variable('user_bias_vector', shape=[self.parameters.user_number])
        self.item_bias_vector = tf.get_variable('item_bias_vector', shape=[self.parameters.item_number])
        print 'global rating'
        print self.parameters.global_rating
        self.global_b = tf.get_variable('global_b', initializer=tf.constant(self.parameters.global_rating, shape=[1], dtype=tf.float32))

    def model_init(self, sess):
        init = tf.initialize_all_variables()
        sess.run(self.word_embedding_matrix.initializer, feed_dict={self.emb_initializer: self.emb})
        sess.run(init)

    def _cnn_process(self, raw_review, users):
        pooled_outputs_i = []
        for i, filter_size in enumerate([3]):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 64, 1, 100]
                self.W = tf.get_variable('W', initializer=tf.truncated_normal(filter_shape, stddev = 0.1))
                self.b = tf.get_variable('b', initializer=tf.constant(0.1, shape=[100]))

                raw_review = tf.reshape(raw_review, [-1, self.max_sentence_word_length, 64, 1])
                self.conv = tf.nn.conv2d(
                    raw_review,
                    self.W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                self.conv_h = tf.nn.relu(tf.nn.bias_add(self.conv, self.b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    self.conv_h,
                    ksize=[1, self.max_sentence_word_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)

        self.num_filters_total = 100 * len([3])
        h_pool_i = tf.concat(pooled_outputs_i, 3)
        h_pool_flat_i = tf.reshape(h_pool_i, [-1, self.max_sentence_length, self.num_filters_total])

        return h_pool_flat_i

    def attentioned_item_review_representation(self, user, item_reviews, item_reviews_length):
        print user
        print item_reviews

        with tf.variable_scope('attention'):
            att = 32
            gamma = 1
            self.attention_w_u = tf.get_variable('attention_w_u', [self.global_dimension, att], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            self.attention_w_r = tf.get_variable('attention_w_r', [self.num_filters_total, att], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            self.attention_w_b = tf.get_variable('attention_w_b', [att], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            self.attention_w_out = tf.get_variable('attention_w_out', [att, 1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            self.attention_b_out = tf.get_variable('attention_b_out', [1], initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

            self.review = tf.reshape(item_reviews, [-1, self.num_filters_total])
            self.review = tf.matmul(self.review, self.attention_w_r)
            self.review = tf.reshape(self.review, [-1, self.max_sentence_length, att])
            self.user_att = tf.expand_dims(tf.matmul(user, self.attention_w_u), 1)
            self.out_att = tf.nn.relu(tf.multiply(self.review, self.user_att) + self.attention_w_b)

            self.out_att = tf.reshape(self.out_att, [-1, att])
            self.out_att = tf.matmul(self.out_att, self.attention_w_out)+self.attention_b_out
            self.out_att = tf.reshape(self.out_att, [-1, self.max_sentence_length, 1])

            attention_weights = tf.nn.softmax(gamma*self.out_att, 1)
            attentioned_review = tf.reduce_sum(tf.multiply(attention_weights, item_reviews), 1)

        return attentioned_review, attention_weights


    def _merge(self, user_embedding, item_embedding, mode='FM', concat=1):
        if mode == 'linear':
            if concat:
                user_item_merge = tf.concat([user_embedding, item_embedding], axis=1)
                result = tf.reduce_sum(user_item_merge, axis=1)
            else:
                user_item_merge = tf.multiply(user_embedding, item_embedding)
                result = tf.reduce_sum(user_item_merge, axis=1)
            return result
        elif mode == 'FM':
            if concat:
                user_item_merge = tf.concat([user_embedding, item_embedding], axis=1)
                FM_W = tf.get_variable('FM_W', initializer=tf.constant(0.1, shape=[2 * self.global_dimension, 1],
                                                                       dtype=tf.float32))
                FM_V = tf.get_variable('FM_V', initializer=tf.truncated_normal([2 * self.global_dimension, 3], stddev=0.1,
                                                                       dtype=tf.float32))
            else:
                user_item_merge = tf.multiply(user_embedding, item_embedding)
                FM_W = tf.get_variable('FM_W', initializer=tf.constant(0.1, shape=[self.global_dimension, 1],
                                                                       dtype=tf.float32))
                FM_V = tf.get_variable('FM_V', initializer=tf.truncated_normal([self.global_dimension, 3], stddev=0.1,
                                                                               dtype=tf.float32))

            self.ll += self.parameters.reg * tf.nn.l2_loss(FM_W) + self.parameters.reg * tf.nn.l2_loss(FM_V)
            fir_inter = tf.squeeze(tf.matmul(user_item_merge, FM_W), [1])
            sec_inter = 0.5 * tf.reduce_sum(
                tf.square(tf.matmul(user_item_merge, FM_V)) - tf.matmul(tf.square(user_item_merge), tf.square(FM_V)), 1)
            result = fir_inter + sec_inter
            return result

        elif mode == 'non-linear':
            if concat:
                user_item_merge = tf.concat([user_embedding, item_embedding], axis=1)
                layer_1_w = tf.get_variable('layer_1_w', initializer=tf.random_uniform([2 * self.global_dimension, self.global_dimension], -1, 1))
                layer_1_b = tf.get_variable('layer_1_b', initializer=tf.random_uniform([self.global_dimension], -1, 1))
            else:
                user_item_merge = tf.multiply(user_embedding, item_embedding)
                layer_1_w = tf.get_variable('layer_1_w', initializer=tf.random_uniform([self.global_dimension, self.global_dimension / 2], -1, 1))
                layer_1_b = tf.get_variable('layer_1_b', initializer=tf.random_uniform([self.global_dimension / 2], -1, 1))

            self.ll += self.parameters.reg * tf.nn.l2_loss(layer_1_w) + self.parameters.reg * tf.nn.l2_loss(layer_1_b)
            layer_1_output = tf.nn.relu(tf.matmul(user_item_merge, layer_1_w) + layer_1_b)
            layer_1_output_drop = tf.nn.dropout(layer_1_output, self.parameters.drop_out_rate)
            result = tf.reduce_sum(layer_1_output_drop, 1)
            return result

    def _GRU(self, inputs, times):
        hidden_size = self.global_dimension
        embed_size = self.global_dimension + self.word_dimension + 5
        batch_size = tf.shape(inputs)[0]
        inputs = tf.unstack(inputs, axis=1)
        times = tf.unstack(times, axis=1)

        with tf.variable_scope("GRU"):
            U_z = tf.get_variable("U_z", shape=(hidden_size, hidden_size))
            U_r = tf.get_variable("U_r", shape=(hidden_size, hidden_size))
            U_s = tf.get_variable("U_s", shape=(hidden_size, hidden_size))
            U = tf.get_variable("U", shape=(hidden_size, hidden_size))

            W_z = tf.get_variable("W_z", shape=(embed_size, hidden_size))
            W_r = tf.get_variable("W_r", shape=(embed_size, hidden_size))
            W_s = tf.get_variable("W_s", shape=(embed_size, hidden_size))
            W = tf.get_variable("W", shape=(embed_size, hidden_size))

            W_s_t = tf.constant(float(self.parameters.lmd), shape=(1, hidden_size))
        
        initial_state = tf.zeros([batch_size, hidden_size])
        pre_h = initial_state

        gru_outputs = []
        for i in range(len(inputs)):
            input = inputs[i]
            time = times[i]
            z_t = tf.sigmoid(tf.matmul(input, W_z) + tf.matmul(pre_h, U_z))
            r_t = tf.sigmoid(tf.matmul(input, W_r) + tf.matmul(pre_h, U_r))
            s_t = tf.sigmoid(tf.matmul(input, W_s) + tf.matmul(pre_h, U_s) +
                             tf.matmul(tf.expand_dims(tf.to_float(time), 1), W_s_t))
            h_t = tf.tanh(tf.matmul(input, W) + tf.multiply(r_t, s_t) * tf.matmul(pre_h, U))
            pre_h = (tf.ones_like(z_t) - tf.multiply(z_t, s_t)) * h_t + tf.multiply(z_t, s_t) * pre_h
            gru_outputs.append(pre_h)
        final_state = pre_h
        return tf.stack(gru_outputs, axis=1), final_state

    def build_loss(self):
        self.logger.info('building model begin ...')
        s = time.time()
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_plh)
        self.previous_items_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.previous_items_plh)
        self.previous_ratings_embedding = tf.nn.embedding_lookup(self.rating_embedding_matrix, tf.to_int32(self.previous_ratings_plh))

        self.current_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_item_plh)
        self.current_input_reviews = tf.nn.embedding_lookup(self.word_embedding_matrix, self.current_input_reviews_plh)
        self.current_input_reviews_users = tf.nn.embedding_lookup(self.user_embedding_matrix, self.current_input_reviews_users_plh)
        self.user_bias = tf.nn.embedding_lookup(self.user_bias_vector, self.user_plh)
        self.current_item_bias = tf.nn.embedding_lookup(self.item_bias_vector, self.current_item_plh)

        with tf.device('/gpu:0'):
            with tf.variable_scope('user_dynamic_rnn'):
                self.previous_preference = tf.concat([self.previous_items_embedding, self.previous_reviews_plh,
                                                      self.previous_ratings_embedding], 2)
                all_outputs, final_state = self._GRU(self.previous_preference, self.previous_times_plh)
                p_batch_size = tf.shape(all_outputs)[0]
                p_index = tf.range(0, p_batch_size) * self.max_interaction_length + (self.previous_lengths_plh - 1)
                self.user_dynamic_output = tf.gather(tf.reshape(all_outputs, [-1, self.global_dimension]), p_index)
                mode = self.parameters.item_review_combine
                k = self.parameters.item_review_combine_c
                if mode == 'add':
                    self.user_match_weight = tf.get_variable('user_match_weight', [self.global_dimension, self.global_dimension], initializer=self.weight_initializer)
                    self.u_dy = tf.matmul(self.user_dynamic_output, self.user_match_weight)
                    self.merged_user = tf.add(k * self.user_embedding, (1 - k) * self.u_dy)
                elif mode == 'mul':
                    self.user_match_weight = tf.get_variable('user_match_weight', [self.global_dimension, self.global_dimension], initializer=self.weight_initializer)
                    self.u_dy = tf.matmul(self.user_dynamic_output, self.user_match_weight)
                    self.merged_user = tf.multiply(k * self.user_embedding, (1 - k) * self.u_dy)
                elif mode == 'concat':
                    self.user_match_weight = tf.get_variable('user_match_weight', [self.global_dimension + self.global_dimension, self.global_dimension], initializer=self.weight_initializer)
                    self.merged_user = tf.matmul(tf.concat([self.user_embedding, k * self.user_dynamic_output], 1), self.user_match_weight)
                self.ll += self.parameters.reg * tf.nn.l2_loss(self.user_match_weight)

            with tf.variable_scope('item_review_cnn'):
                self.current_input_reviews_cnn = self._cnn_process(self.current_input_reviews, self.current_input_reviews_users)
                self.item_review_representaion, self.item_review_attention = \
                    self.attentioned_item_review_representation(self.user_dynamic_output, self.current_input_reviews_cnn,
                                                                self.current_input_reviews_length_plh)

                mode = self.parameters.item_review_combine
                k = self.parameters.item_review_combine_c
                if mode == 'add':
                    self.match_weight = tf.get_variable('match_weight', [self.num_filters_total, self.global_dimension], initializer=self.weight_initializer)
                    self.i_dy = tf.matmul(self.item_review_representaion, self.match_weight)
                    self.merged_item = tf.add(k * self.current_item_embedding, (1 - k) * self.i_dy)
                elif mode == 'mul':
                    self.match_weight = tf.get_variable('match_weight', [self.num_filters_total, self.global_dimension], initializer=self.weight_initializer)
                    self.i_dy = tf.matmul(self.item_review_representaion, self.match_weight)
                    self.merged_item = tf.multiply(k * self.current_item_embedding, (1 - k) * self.i_dy)
                elif mode == 'concat':
                    self.match_weight = tf.get_variable('match_weight', [self.num_filters_total + self.global_dimension, self.global_dimension], initializer=self.weight_initializer)
                    self.merged_item = tf.matmul(tf.concat([self.current_item_embedding, k * self.item_review_representaion], 1), self.match_weight)
                self.ll += self.parameters.reg * tf.nn.l2_loss(self.match_weight)

            with tf.variable_scope('final_prediction'):
                self.merged_user = tf.nn.dropout(self.merged_user, self.parameters.drop_out_rate)
                self.merged_item = tf.nn.dropout(self.merged_item, self.parameters.drop_out_rate)
                self.pre_rating = self._merge(self.merged_user, self.merged_item, self.parameters.merge, self.parameters.concat)
                self.pre_rating += self.user_bias + self.current_item_bias + self.global_b
                self.mse_sum = tf.nn.l2_loss(self.current_rating_plh - self.pre_rating) * 2
                self.total_loss = tf.nn.l2_loss(self.current_rating_plh - self.pre_rating) + self.ll

            time_consuming = str(time.time() - s)
            self.logger.info('building model end ... time consuming: ' + time_consuming)

    def get_train_op(self):
        self.total_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        #self.total_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        all_vars = tf.trainable_variables()
        vars = [var for var in all_vars if var.name.split('/')[0] == self.namespace]
        gvs = self.total_optimizer.compute_gradients(self.total_loss, vars)
        print gvs
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = self.total_optimizer.apply_gradients(capped_gvs)
        return self.train_op

    def get_attention_weight(self):
        return self.item_review_attention

    def get_rating_prediction(self):
        return self.pre_rating

    def get_loss(self):
        return self.total_loss

    def get_mse_sum(self):
        return self.mse_sum





