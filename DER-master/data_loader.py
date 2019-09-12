import numpy as np
import pandas as pd
import pickle
import time
import os
import itertools


class DataLoader():

    def __init__(self, args):
        np.random.seed(0)
        print 'init'
        self.logger = args.logger
        self.train_file_path = os.path.join(args.root_path, args.input_data_type, 'train_ided_whole_data')
        self.test_file_path = os.path.join(args.root_path, args.input_data_type, 'test_ided_whole_data')
        self.validation_file_path = os.path.join(args.root_path, args.input_data_type, 'validation_ided_whole_data')
        self.train_validation_file_path = os.path.join(args.root_path, args.input_data_type, 'train_validation_ided_whole_data')

        self.id_user_dict_path = os.path.join(args.root_path, args.input_data_type, 'id_user_dict')
        self.id_item_dict_path = os.path.join(args.root_path, args.input_data_type, 'id_item_dict')
        self.id_word_dict_path = os.path.join(args.root_path, args.input_data_type, 'id_word_dict')
        self.item_reviews_path = os.path.join(args.root_path, args.input_data_type, 'item_reviews')
        self.item_real_reviews_path = os.path.join(args.root_path, args.input_data_type, 'item_real_reviews')
        self.user_item_review_path = os.path.join(args.root_path, args.input_data_type, 'user_item_review')
        self.train_user_purchsed_items_dict_path = os.path.join(args.root_path, args.input_data_type, 'train_user_purchased_items')
        self.validation_user_purchsed_items_dict_path = os.path.join(args.root_path, args.input_data_type, 'validation_user_purchased_items')
        self.test_user_purchsed_items_dict_path = os.path.join(args.root_path, args.input_data_type, 'test_user_purchased_items')
        self.data_statistics_path = os.path.join(args.root_path, args.input_data_type, 'data_statistics')

        self.logger.info('loading dicts begin ...')
        s = time.time()
        self.train_data = pd.read_csv(self.train_file_path, header=None, dtype='str')
        self.validation_data = pd.read_csv(self.validation_file_path, header=None, dtype='str')
        self.test_data = pd.read_csv(self.test_file_path, header=None, dtype='str')
        self.train_validation_data = pd.read_csv(self.train_validation_file_path, header=None, dtype='str')

        self.id_user_dict = pickle.load(open(self.id_user_dict_path, 'rb'))
        self.id_item_dict = pickle.load(open(self.id_item_dict_path, 'rb'))
        self.id_word_dict = pickle.load(open(self.id_word_dict_path, 'rb'))
        self.item_reviews_dict = pickle.load(open(self.item_reviews_path, 'rb'))
        self.item_real_reviews_dict = pickle.load(open(self.item_real_reviews_path, 'rb'))
        self.user_item_review_dict = pickle.load(open(self.user_item_review_path, 'rb'))

        self.train_user_purchsed_items_dict = pickle.load(open(self.train_user_purchsed_items_dict_path, 'rb'))
        self.validation_user_purchsed_items_dict = pickle.load(open(self.validation_user_purchsed_items_dict_path, 'rb'))
        self.test_user_purchsed_items_dict = pickle.load(open(self.test_user_purchsed_items_dict_path, 'rb'))
        self.data_statistics = pickle.load(open(self.data_statistics_path, 'rb'))
        time_consuming = str(time.time() - s)
        self.logger.info('loading dicts end ... time consuming: ' + time_consuming)

        self.max_interaction_length = self.data_statistics['max_interaction_length']
        self.interaction_num = self.data_statistics['interaction_num']

        self.max_sentence_length = self.data_statistics['max_sentence_length']
        self.max_sentence_length = 30
        self.max_sentence_word_length = self.data_statistics['max_sentence_word_length']
        self.max_sentence_word_length = 10
        self.time_bin_number = self.data_statistics['time_bin_number']
        self.user_num = self.data_statistics['user_num']
        self.item_num = self.data_statistics['item_num']
        self.word_num = self.data_statistics['word_num']
        self.global_rating = 0.0

        # all data statistics
        self.logger.info('user number: ' + str(self.user_num))
        self.logger.info('item number: ' + str(self.item_num))
        self.logger.info('word number: ' + str(self.word_num))
        self.logger.info('interaction number: ' + str(self.interaction_num))
        self.logger.info('max interaction length: ' + str(self.max_interaction_length))
        self.logger.info('max sentence length: ' + str(self.max_sentence_length))
        self.logger.info('max sentence word length: ' + str(self.max_sentence_word_length))
        self.logger.info('sparsity: ' + str(int(self.interaction_num) / (float(self.user_num)*float(self.item_num))))

        # aligned training data
        self.train_batch_id = 0
        self.train_records_num = 0

        self.train_users = []
        self.train_previous_items = []
        self.train_previous_times = []
        self.train_previous_ratings = []
        self.train_previous_reviews = []
        self.train_previous_lengths = []
        self.train_current_items = []
        self.train_current_ratings = []
        self.train_current_input_reviews = []
        self.train_current_input_reviews_users = []
        self.train_current_input_reviews_length = []
        self.train_current_output_review = []

        # aligned validation data
        self.validation_batch_id = 0
        self.validation_records_num = 0

        self.validation_users = []
        self.validation_previous_items = []
        self.validation_previous_times = []
        self.validation_previous_ratings = []
        self.validation_previous_reviews = []
        self.validation_previous_lengths = []
        self.validation_current_items = []
        self.validation_current_ratings = []
        self.validation_current_input_reviews = []
        self.validation_current_input_reviews_users = []
        self.validation_current_input_reviews_length = []
        self.validation_current_output_review = []

        # aligned testing data
        self.test_batch_id = 0
        self.test_records_num = 0

        self.test_users = []
        self.test_previous_items = []
        self.test_previous_times = []
        self.test_previous_ratings = []
        self.test_previous_reviews = []
        self.test_previous_lengths = []
        self.test_current_items = []
        self.test_current_ratings = []
        self.test_current_input_reviews = []
        self.test_current_input_reviews_users = []
        self.test_current_input_reviews_length = []
        self.test_current_output_review = []

        # aligned train_validation data
        self.train_validation_batch_id = 0
        self.train_validation_records_num = 0

        self.train_validation_users = []
        self.train_validation_previous_items = []
        self.train_validation_previous_times = []
        self.train_validation_previous_ratings = []
        self.train_validation_previous_reviews = []
        self.train_validation_previous_lengths = []
        self.train_validation_current_items = []
        self.train_validation_current_ratings = []
        self.train_validation_current_input_reviews = []
        self.train_validation_current_input_reviews_users = []
        self.train_validation_current_input_reviews_length = []
        self.train_validation_current_output_review = []

    def _pad_sequence(self, mask_value, max_length, input_sequence):
        real_length = len(input_sequence)
        if real_length < max_length:
            output_sequence = input_sequence + [mask_value] * (max_length - real_length)
            length = real_length
        else:
            output_sequence = input_sequence[:max_length]
            length = max_length
        return output_sequence, length

    def gen_data(self, mode='train'):
        self.logger.info('generating ' + mode + ' data...')
        data = eval('self.' + mode + '_data')
        for line in data.values:
            list_line = line[0].split('&&')
            user = int(list_line[0])

            items = [int(i.split('||')[0]) for i in list_line[1].split('()')]
            paded_items, length = self._pad_sequence(0, self.max_interaction_length, items)

            ratings = [float(i.split('||')[1]) for i in list_line[1].split('()')]
            paded_ratings, length = self._pad_sequence(0.0, self.max_interaction_length, ratings)

            reviews = [self.user_item_review_dict[list_line[0]+'@'+str(j)] for j in items]
            reviews = list(itertools.chain.from_iterable(reviews))
            reviews = [i.tolist() for i in reviews]
            paded_reviews, length = self._pad_sequence([0.0]*64, self.max_interaction_length, reviews)

            times = [int(i.split('||')[3]) for i in list_line[1].split('()')]
            paded_times, length = self._pad_sequence(0.0, self.max_interaction_length, times)

            c_item = int(list_line[2].split('||')[0])
            c_rating = float(list_line[2].split('||')[1])
            if mode == 'train':
                self.global_rating += c_rating
            c_output_review = [int(j) for j in list_line[2].split('||')[2].split('::')]
            c_input_reviews = [i[2] for i in self.item_reviews_dict[str(c_item)] if i[0] != list_line[0]]
            c_input_reviews_users = [i[0] for i in self.item_reviews_dict[str(c_item)] if i[0] != list_line[0]]
            tim = [len(i) for i in c_input_reviews]
            c_input_reviews_users = [[int(c_input_reviews_users[i])]*tim[i] for i in range(len(c_input_reviews_users))]
            c_input_reviews = list(itertools.chain.from_iterable(c_input_reviews))
            c_input_reviews_users = list(itertools.chain.from_iterable(c_input_reviews_users))
            c_input_reviews = [self._pad_sequence(0, self.max_sentence_word_length, i)[0] for i in c_input_reviews]

            c_input_reviews, input_length = self._pad_sequence([0]*self.max_sentence_word_length, self.max_sentence_length, c_input_reviews)
            c_input_reviews_users, input_length_users = self._pad_sequence(0, self.max_sentence_length, c_input_reviews_users)

            #print len(c_input_reviews)
            #print len(c_input_reviews[0])
            #print len(c_input_reviews[0][0])

            #print list
            #print user
            #print paded_items
            #print paded_ratings
            #print paded_reviews
            #print paded_times
            #print length
            #print c_item
            #print c_rating
            #print c_input_reviews[0]
            #print c_input_reviews[0][0]
            #print input_length
            #print c_output_review
            #print c_input_reviews
            #print c_input_reviews_users
            #print len(c_input_reviews)
            #print len(c_input_reviews_users)
            #raw_input()

            eval('self.' + mode + '_users').append(user)
            eval('self.' + mode + '_previous_items').append(paded_items)
            eval('self.' + mode + '_previous_times').append(paded_times)
            eval('self.' + mode + '_previous_reviews').append(paded_reviews)
            eval('self.' + mode + '_previous_ratings').append(paded_ratings)
            eval('self.' + mode + '_previous_lengths').append(length)
            eval('self.' + mode + '_current_items').append(c_item)
            eval('self.' + mode + '_current_ratings').append(c_rating)
            eval('self.' + mode + '_current_input_reviews').append(c_input_reviews)
            eval('self.' + mode + '_current_input_reviews_users').append(c_input_reviews_users)
            eval('self.' + mode + '_current_input_reviews_length').append(input_length)
            eval('self.' + mode + '_current_output_review').append(c_output_review)

            if mode == 'train':
                self.train_records_num += 1
            elif mode == 'validation':
                self.validation_records_num += 1
            elif mode == 'test':
                self.test_records_num += 1
            else:
                self.train_validation_records_num += 1
        print mode +' number:' + str(eval('self.' + mode + '_records_num'))
        if mode == 'train':
            self.global_rating /= self.train_records_num

    def shuffle(self):
        self.logger.info('shuffle ...')
        self.index = np.array(range(self.train_records_num))
        np.random.shuffle(self.index)

        self.train_users = list(np.array(self.train_users)[self.index])
        self.train_previous_items = list(np.array(self.train_previous_items)[self.index])
        self.train_previous_times = list(np.array(self.train_previous_times)[self.index])
        self.train_previous_reviews = list(np.array(self.train_previous_reviews)[self.index])
        self.train_previous_ratings = list(np.array(self.train_previous_ratings)[self.index])
        self.train_previous_lengths = list(np.array(self.train_previous_lengths)[self.index])
        self.train_current_items = list(np.array(self.train_current_items)[self.index])
        self.train_current_ratings = list(np.array(self.train_current_ratings)[self.index])
        self.train_current_input_reviews = list(np.array(self.train_current_input_reviews)[self.index])
        self.train_current_input_reviews_users = list(np.array(self.train_current_input_reviews_users)[self.index])
        self.train_current_output_review = list(np.array(self.train_current_output_review)[self.index])
        self.train_current_input_reviews_length = list(np.array(self.train_current_input_reviews_length)[self.index])

    def gen_batch_data(self, start, end, mode='train'):
        batch_users = eval('self.' + mode + '_users')[start:end]
        batch_previous_items = eval('self.' + mode + '_previous_items')[start:end]
        batch_previous_times = eval('self.' + mode + '_previous_times')[start:end]

        batch_previous_reviews = eval('self.' + mode + '_previous_reviews')[start:end]
        batch_previous_ratings = eval('self.' + mode + '_previous_ratings')[start:end]
        batch_previous_lengths = eval('self.' + mode + '_previous_lengths')[start:end]

        batch_current_items = eval('self.' + mode + '_current_items')[start:end]
        batch_current_ratings = eval('self.' + mode + '_current_ratings')[start:end]
        batch_current_input_reviews = eval('self.' + mode + '_current_input_reviews')[start:end]
        batch_current_input_reviews_users = eval('self.' + mode + '_current_input_reviews_users')[start:end]
        batch_current_output_review = eval('self.' + mode + '_current_output_review')[start:end]
        batch_current_input_reviews_length = eval('self.' + mode + '_current_input_reviews_length')[start:end]


        return batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
               batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
               batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length

