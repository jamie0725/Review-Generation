from data_loader import DataLoader
from model.DER import *
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import os
import itertools
from compiler.ast import flatten

class Solver:
    def __init__(self, args, id):
        self.args = args
        self.data_loader = DataLoader(args)
        self.data_loader.gen_data('train')
        self.data_loader.gen_data('validation')
        self.data_loader.gen_data('test')
        self.data_loader.gen_data('train_validation')
        self.model_args = self.args

        self.model_args.user_number = self.data_loader.user_num
        self.model_args.item_number = self.data_loader.item_num
        self.model_args.word_number = self.data_loader.word_num
        self.model_args.max_interaction_length = self.data_loader.max_interaction_length
        self.model_args.max_sentence_length = self.data_loader.max_sentence_length
        self.model_args.max_sentence_word_length = self.data_loader.max_sentence_word_length
        self.model_args.time_bin_number = self.data_loader.time_bin_number
        self.model_args.global_rating = self.data_loader.global_rating
        self.experiment_id = id

        self.model = DER(self.model_args)
        self.model.build_loss()
        print self.model_args
        # Get loss, evaluation, operation, rating and attention prediction.
        self.loss = self.model.get_loss()
        self.evaluation_mse_sum = self.model.get_mse_sum()
        self.op = self.model.get_train_op()
        self.rating_prediction = self.model.get_rating_prediction()
        self.attention_prediction = self.model.get_attention_weight()
        self.item_reviews_dict = self.data_loader.item_reviews_dict
        self.item_real_reviews_dict = self.data_loader.item_real_reviews_dict

    def save_parameters(self):
        path = os.path.join(self.args.output_path, str(self.experiment_id))
        if not os.path.exists(path):
            os.makedirs(path)
        self.args.logger.info('saving parameters ...')

        arg_dict = dict()
        for k, v in self.model_args.__dict__.items():
            if k != 'logger':
                arg_dict[k] = v

        pickle.dump(arg_dict, open(os.path.join(self.args.output_path, str(self.experiment_id), 'model_args'), 'wb'))
        t = pd.DataFrame(self.train_rmse_vs_epoch)
        t.to_csv(os.path.join(self.args.output_path, str(self.experiment_id), 'train_rmse_vs_epoch'), index=False, header=None)
        t = pd.DataFrame(self.validation_rmse_vs_epoch)
        t.to_csv(os.path.join(self.args.output_path, str(self.experiment_id), 'validation_rmse_vs_epoch'), index=False, header=None)


    def save_attention(self):
        self.args.logger.info('saving attention ...')
        self.step = 0
        self.result = []
        max_step = self.data_loader.test_records_num / self.args.batch_size
        while self.step <= max_step:
            if (self.step + 1) * self.args.batch_size > self.data_loader.test_records_num:
                b = self.data_loader.test_records_num - self.step * self.args.batch_size
            else:
                b = self.args.batch_size

            start = self.step * self.args.batch_size
            end = start + b
            batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
            batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
            batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length \
                = self.data_loader.gen_batch_data(start, end, 'test')
            self.step += 1

            att, a, b, c = self.sess.run([self.model.item_review_attention, self.model.out_att, self.model.review, self.model.user_att],
                                       feed_dict={self.model.user_plh: batch_users,
                                                  self.model.previous_items_plh: batch_previous_items,
                                                  self.model.previous_times_plh: batch_previous_times,
                                                  self.model.previous_reviews_plh: batch_previous_reviews,
                                                  self.model.previous_ratings_plh: batch_previous_ratings,
                                                  self.model.previous_lengths_plh: batch_previous_lengths,
                                                  self.model.current_item_plh: batch_current_items,
                                                  self.model.current_rating_plh: batch_current_ratings,
                                                  self.model.current_input_reviews_plh: batch_current_input_reviews,
                                                  self.model.current_input_reviews_users_plh: batch_current_input_reviews_users,
                                                  self.model.current_input_reviews_length_plh: batch_current_input_reviews_length}
                                       )



            '''
            print np.array(att).shape
            print att[0]
            print att[1]
            print np.array(a).shape
            print a[0]
            print np.array(b).shape
            print b[0]
            print np.array(c).shape
            print c
            print d
            raw_input()
            '''


            for i in range(len(att)):
                user = batch_users[i]
                item = batch_current_items[i]
                attention = '@@@'.join([str(k[0]) for k in att[i]])
                reviews = [j[1] for j in self.item_real_reviews_dict[str(batch_current_items[i])] if j[0] != str(batch_users[i])]
                reviews = list(itertools.chain.from_iterable(reviews))[:self.model_args.max_sentence_length]
                reviews = '@@@'.join(reviews)
                line = str(user) + '||' + str(item) + '||' + attention + '||' + reviews
                self.result.append(line)

        path = os.path.join(self.args.output_path, str(self.experiment_id))
        if not os.path.exists(path):
            os.makedirs(path)
        t = pd.DataFrame(self.result)
        t.to_csv(os.path.join(self.args.output_path, str(self.experiment_id), 'attention_results'), index=False, header=None)

    def run(self):
        self.args.logger.info('running ...')
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True
        self.min_rmse = 1000000
        self.test_min_rmse = 1000000
        self.train_rmse_vs_epoch = []
        self.validation_rmse_vs_epoch = []
        self.test_rmse_vs_epoch = []

        with tf.Session(config=conf) as self.sess:
            self.model.model_init(self.sess)
            for iter in range(self.args.epoch):
                self.args.logger.info('*************************************')
                self.args.logger.info('epoch: ' + str(iter) + ' begin.')
                s = time.time()
                self.data_loader.shuffle()
                t_loss = 0.0
                self.step = 0
                if self.model_args.mode == 'validation':
                    max_step = self.data_loader.train_records_num/self.args.batch_size
                    record_number = self.data_loader.train_records_num
                else:
                    max_step = self.data_loader.train_validation_records_num / self.args.batch_size
                    record_number = self.data_loader.train_validation_records_num
                while self.step <= max_step:
                    if (self.step+1) * self.args.batch_size > record_number:
                        b = record_number - self.step * self.args.batch_size
                    else:
                        b = self.args.batch_size

                    start = self.step * self.args.batch_size
                    end = start + b
                    if end > start:
                        if self.model_args.mode == 'validation':
                            batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
                            batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
                            batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length \
                                = self.data_loader.gen_batch_data(start, end, 'train')
                        else:
                            batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
                            batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
                            batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length \
                                = self.data_loader.gen_batch_data(start, end, 'train_validation')

                        self.sess.run(self.op, feed_dict={self.model.user_plh: batch_users,
                                                          self.model.previous_items_plh: batch_previous_items,
                                                          self.model.previous_times_plh: batch_previous_times,
                                                          self.model.previous_reviews_plh: batch_previous_reviews,
                                                          self.model.previous_ratings_plh: batch_previous_ratings,
                                                          self.model.previous_lengths_plh: batch_previous_lengths,
                                                          self.model.current_item_plh: batch_current_items,
                                                          self.model.current_rating_plh: batch_current_ratings,
                                                          self.model.current_input_reviews_plh: batch_current_input_reviews,
                                                          self.model.current_input_reviews_users_plh: batch_current_input_reviews_users,
                                                          self.model.current_input_reviews_length_plh: batch_current_input_reviews_length}
                                      )
                        self.step += 1


                time_consuming = str(time.time() - s)
                self.args.logger.info('epoch: ' + str(iter) + ' end. time consuming: ' + time_consuming)


                self.args.logger.info('epoch: ' + str(iter) + ' eval begin.')
                s = time.time()
                mse_sum = 0.0
                self.step = 0
                max_step = self.data_loader.train_records_num / self.args.batch_size
                while self.step <= max_step:
                    if (self.step+1) * self.args.batch_size > self.data_loader.train_records_num:
                        b = self.data_loader.train_records_num - self.step * self.args.batch_size
                    else:
                        b = self.args.batch_size
                    start = self.step * self.args.batch_size
                    end = start + b
                    if end > start:
                        batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
                        batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
                        batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length \
                            = self.data_loader.gen_batch_data(start, end, 'train')
                        loss = self.sess.run(self.evaluation_mse_sum,
                                             feed_dict={self.model.user_plh: batch_users,
                                                        self.model.previous_items_plh: batch_previous_items,
                                                        self.model.previous_times_plh: batch_previous_times,
                                                        self.model.previous_reviews_plh: batch_previous_reviews,
                                                        self.model.previous_ratings_plh: batch_previous_ratings,
                                                        self.model.previous_lengths_plh: batch_previous_lengths,
                                                        self.model.current_item_plh: batch_current_items,
                                                        self.model.current_rating_plh: batch_current_ratings,
                                                        self.model.current_input_reviews_plh: batch_current_input_reviews,
                                                        self.model.current_input_reviews_users_plh: batch_current_input_reviews_users,
                                                        self.model.current_input_reviews_length_plh: batch_current_input_reviews_length}
                                             )
                        mse_sum += np.array(loss).sum()
                        self.step += 1

                rmse = np.sqrt(mse_sum / self.data_loader.train_records_num)
                self.train_rmse_vs_epoch.append(rmse)
                self.args.logger.info('epoch: ' + str(iter) + ' training loss: ' + str(rmse))

                mse_sum = 0.0
                self.step = 0
                max_step = self.data_loader.validation_records_num / self.args.batch_size
                while self.step <= max_step:
                    if (self.step + 1) * self.args.batch_size > self.data_loader.validation_records_num:
                        b = self.data_loader.validation_records_num - self.step * self.args.batch_size
                    else:
                        b = self.args.batch_size

                    start = self.step * self.args.batch_size
                    end = start + b
                    if end > start:
                        batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
                        batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
                        batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length \
                            = self.data_loader.gen_batch_data(start, end, 'validation')
                        error = self.sess.run(self.evaluation_mse_sum, feed_dict={self.model.user_plh: batch_users,
                                                                                  self.model.previous_items_plh: batch_previous_items,
                                                                                  self.model.previous_times_plh: batch_previous_times,
                                                                                  self.model.previous_reviews_plh: batch_previous_reviews,
                                                                                  self.model.previous_ratings_plh: batch_previous_ratings,
                                                                                  self.model.previous_lengths_plh: batch_previous_lengths,
                                                                                  self.model.current_item_plh: batch_current_items,
                                                                                  self.model.current_rating_plh: batch_current_ratings,
                                                                                  self.model.current_input_reviews_plh: batch_current_input_reviews,
                                                                                  self.model.current_input_reviews_users_plh: batch_current_input_reviews_users,
                                                                                  self.model.current_input_reviews_length_plh: batch_current_input_reviews_length}
                                              )
                        mse_sum += np.array(error).sum()
                        self.step += 1

                rmse = np.sqrt(mse_sum/self.data_loader.validation_records_num)
                self.validation_rmse_vs_epoch.append(rmse)
                if rmse < self.min_rmse:
                    self.min_rmse = rmse
                    #self.save_attention()
                self.args.logger.info('epoch: ' + str(iter) + ' validation rmse: ' + str(rmse))
                self.args.logger.info('current best rmse: ' + str(self.min_rmse))

                mse_sum = 0.0
                self.step = 0
                max_step = self.data_loader.test_records_num / self.args.batch_size
                while self.step <= max_step:
                    if (self.step + 1) * self.args.batch_size > self.data_loader.test_records_num:
                        b = self.data_loader.test_records_num - self.step * self.args.batch_size
                    else:
                        b = self.args.batch_size

                    start = self.step * self.args.batch_size
                    end = start + b
                    if end > start:
                        batch_users, batch_previous_items, batch_previous_times, batch_previous_reviews, \
                        batch_previous_ratings, batch_previous_lengths, batch_current_items, batch_current_ratings, \
                        batch_current_input_reviews, batch_current_input_reviews_users, batch_current_output_review, batch_current_input_reviews_length \
                            = self.data_loader.gen_batch_data(start, end, 'test')
                        error = self.sess.run(self.evaluation_mse_sum, feed_dict={self.model.user_plh: batch_users,
                                                                                  self.model.previous_items_plh: batch_previous_items,
                                                                                  self.model.previous_times_plh: batch_previous_times,
                                                                                  self.model.previous_reviews_plh: batch_previous_reviews,
                                                                                  self.model.previous_ratings_plh: batch_previous_ratings,
                                                                                  self.model.previous_lengths_plh: batch_previous_lengths,
                                                                                  self.model.current_item_plh: batch_current_items,
                                                                                  self.model.current_rating_plh: batch_current_ratings,
                                                                                  self.model.current_input_reviews_plh: batch_current_input_reviews,
                                                                                  self.model.current_input_reviews_users_plh: batch_current_input_reviews_users,
                                                                                  self.model.current_input_reviews_length_plh: batch_current_input_reviews_length}
                                              )
                        # print error
                        mse_sum += np.array(error).sum()
                        self.step += 1
                rmse = np.sqrt(mse_sum / self.data_loader.test_records_num)
                self.test_rmse_vs_epoch.append(rmse)
                if rmse < self.test_min_rmse:
                    self.test_min_rmse = rmse
                    self.save_attention()

                self.args.logger.info('epoch: ' + str(iter) + ' test rmse: ' + str(rmse))
                self.args.logger.info('current best test rmse: ' + str(self.test_min_rmse))
                time_consuming = str(time.time() - s)
                self.args.logger.info('epoch: ' + str(iter) + ' eval end. time consuming: ' + time_consuming)
                self.args.logger.info('*************************************')

            self.save_parameters()
            self.sess.close()
            if self.model_args.mode == 'validation':
                return self.min_rmse
            else:
                return self.test_min_rmse


