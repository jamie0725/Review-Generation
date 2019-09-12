import pickle
import pandas as pd
import os
import numpy as np


experiment_id = 0
attention_path = os.path.join('./results', str(experiment_id), 'attention_results')
attention = pd.read_csv(attention_path, header=None, dtype='str')

path = './data/der_data/auto/'
id_user_dict = {v:k for k,v in pickle.load(open(path+'user_id_dict', 'rb')).items()}


id_item_dict = {v:k for k,v in pickle.load(open(path+'item_id_dict', 'rb')).items()}
id_word_dict = pickle.load(open(path+'id_word_dict', 'rb'))
user_reviews_dict = pickle.load(open(path+'user_reviews', 'rb'))
item_reviews_dict = pickle.load(open(path+'item_real_reviews', 'rb'))



train_rmse_path = os.path.join('./results', str(experiment_id), 'train_rmse_vs_epoch')
train_rmse = pd.read_csv(train_rmse_path, header=None, dtype='str')
validation_rmse_path = os.path.join('./results', str(experiment_id), 'validation_rmse_vs_epoch')
validation_rmse = pd.read_csv(validation_rmse_path, header=None, dtype='str')
parameters = pickle.load(open(os.path.join('./results', str(experiment_id), 'model_args'), 'rb'))

num = 0

final_result = dict()

for line in attention.values:
    num += 1
    l_list = line[0].split('||')
    user = l_list[0]
    item = l_list[1]

    sentences = np.array(l_list[3].split('@@@'))
    attention = np.array([float(i) for i in l_list[2].split('@@@')])[:len(sentences)]

    max_index = np.argsort(attention)[::-1][:5]
    max_index = [i for i in max_index if i in range(len(sentences))]

    #print attention[max_index]
    #print sentences[max_index]
    user_r = user_reviews_dict[id_user_dict[user]]

    user_r_index = [i.split('||')[1] for i in user_r]
    user_r_content = [i.split('||')[0] for i in user_r]
    user_r_item = [i.split('||')[2] for i in user_r]

    sort_index = np.argsort(user_r_index)
    previous_time = list(np.array(user_r_index)[sort_index])
    previous_reviews = list(np.array(user_r_content)[sort_index])

    record = str(user) + '||' + '@@@'.join(previous_time) + '||' + '@@@'.join(previous_reviews) \
             + '||' + '@@@'.join([str(i) for i in attention[max_index]]) + '||' + '@@@'.join(sentences[max_index])

    if item not in final_result.keys():
        final_result[item] = [record]
    else:
        final_result[item].append(record)


for k,v in final_result.items():
    print 'item reviews:'
    print str(k)
    print item_reviews_dict[k]
    print 'user attention for this item:'
    for line in v:
        line_list = line.split('||')
        print 'user:'
        print line_list[0]

        print 'previous review'
        for p in range(len(line_list[1].split('@@@')[:-1])):
            print 'time:'
            print line_list[1].split('@@@')[:-1][p]
            print 'content:'
            print line_list[2].split('@@@')[:-1][p]
        print '----------------------------'
        print 'current time:'
        print line_list[1].split('@@@')[-1]
        print 'current review:'
        print line_list[2].split('@@@')[-1]
        print '----------------------------'

        print 'attention:'
        print line_list[3].split('@@@')
        print 'sentences:'
        print line_list[4].split('@@@')
        print '&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
    if int(k) == 669:
        raw_input()

    print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'







train = []
validation = []

for line in train_rmse.values:
    train.append(float(line[0]))

for line in validation_rmse.values:
    validation.append(float(line[0]))

print train
print validation