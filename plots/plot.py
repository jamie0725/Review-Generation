import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Some global paarameters.
NUM_EPOCHS = 20
MRG_OUT_FILE = '../mrg/log_train.out'
MRG_GRU_OUT_FILE = '../mrg-gru/log_train.out'
MRG_GRU_ATTN_OUT_FILE = '../mrg-gru-attn/log_train.out'
RATING_LOSS_FIG_SAVE = './rating_loss.pdf'
REVIEW_LOSS_FIG_SAVE = './review_loss.pdf'
TRAIN_LOSS_PATTERN = re.compile('Training.*219/219.*rating_loss=(?P<rate_loss>[.\d]+), review_loss=(?P<review_loss>[.\d]+)')


def extract_from_file(file_loc):
    rate_losses = []
    review_losses = []
    file = open(file_loc, "r", encoding='utf-8')
    for line in file:
        match = re.search(TRAIN_LOSS_PATTERN, line)
        if match:
            rate_loss = match['rate_loss']
            review_loss = match['review_loss']
            rate_losses.append(float(rate_loss))
            review_losses.append(float(review_loss))
    # assert len(review_losses) == NUM_EPOCHS
    # assert len(rate_losses) == NUM_EPOCHS
    return rate_losses, review_losses


if __name__ == "__main__":
    # Read from logging files.
    mrg_rating_loss, mrg_review_loss = extract_from_file(MRG_OUT_FILE)
    mrg_gru_rating_loss, mrg_gru_review_loss = extract_from_file(MRG_GRU_OUT_FILE)
    mrg_gru_attn_rating_loss, mrg_gru_attn_review_loss = extract_from_file(MRG_GRU_ATTN_OUT_FILE)

    len_mrg = len(mrg_rating_loss)
    len_gru = len(mrg_gru_rating_loss)
    len_attn = len(mrg_gru_attn_rating_loss)

    # Plotting.
    fig_review, ax_review = plt.subplots()
    plt.plot(np.arange(1, len_mrg+1, 1), mrg_review_loss, 'r', np.arange(1, len_gru+1, 1),
             mrg_gru_review_loss, 'b', np.arange(1, len_attn+1, 1), mrg_gru_attn_review_loss, 'g')
    plt.xticks(np.arange(1, NUM_EPOCHS+1, 3))
    plt.legend(('MRG', 'MRG-PE', 'MRG-PEA'), shadow=True)
    plt.xlabel('Epoch')
    plt.ylabel('Review Loss')
    plt.show()
    fig_review.savefig(REVIEW_LOSS_FIG_SAVE, format='pdf')

    fig_rating, ax_rating = plt.subplots()
    plt.plot(np.arange(1, len_mrg+1, 1), mrg_rating_loss, 'r', np.arange(1, len_gru+1, 1),
             mrg_gru_rating_loss, 'b', np.arange(1, len_attn+1, 1), mrg_gru_attn_rating_loss, 'g')
    plt.xticks(np.arange(1, NUM_EPOCHS+1, 3))
    plt.legend(('MRG', 'MRG-PE', 'MRG-PEA'), shadow=True)
    plt.xlabel('Epoch')
    plt.ylabel('Rating Loss')
    plt.show()
    fig_rating.savefig(RATING_LOSS_FIG_SAVE, format='pdf')
