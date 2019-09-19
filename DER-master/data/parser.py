# In[1]:


import json
import csv
import pickle
import re
import numpy as np

from collections import defaultdict
from gensim.models import Word2Vec
from gensim.scripts import word2vec2tensor

filename_input = 'reviews_Automotive_5.json'
# Load the raw data
raw_data = [json.loads(line) for line in open(filename_input, 'r')]


# In[2]:


# Generate id_user_dict/id_item_dict/id_word_dict
users = set()
items = set()
vocab = set()
full_raw_text = []
max_interaction_length = 0
for line in raw_data:
    UserID = line['reviewerID']
    ItemID = line['asin']
    sentence = re.sub(r'[^\w\s]|\d','',line['reviewText'].lower())
    words = [word for word in  sentence.split()]
    full_raw_text.append(words)
    if len(words) > max_interaction_length:
        max_interaction_length = len(words)
    users.add(UserID)
    items.add(ItemID)
    for word in words:
        vocab.add(word)
id_user_dict = dict(zip(range(len(users)), users))
id_item_dict = dict(zip(range(len(items)), items))
id_word_dict = dict(zip(range(len(vocab)), vocab))
word_id_dict = dict(zip(vocab,range(len(vocab))))

# In[3]:


print('length of id_user_dict:',len(id_user_dict))
print('length of id_item_dict:',len(id_item_dict))
print('length of id_word_dict:',len(id_word_dict))


# In[4]:


def get_key_by_value(value,D):
    id = list(D.values()).index(value)
    return list(D.keys())[id]

def review2id(review):
    review_text = re.sub(r'[^\w\s]|\d','',review['reviewText'].lower())
    sentence_ids = [word_id_dict[word] for word in review_text.split()]
    return np.array(sentence_ids)

# In[5]:


"""
item_reviews
    Key: ItemID 
    Value: All the user reviews received by the Item
user_item_review
    Key: UserID@ItemID 
    Value: review from the user to the item
user_purchased_items
    Key: UserID
    Value: list of purchasedID
"""
item_real_reviews = defaultdict(list)
item_reviews = defaultdict(list)
user_item_real_review = defaultdict(list)
user_item_review = defaultdict(list)
user_purchased_items = defaultdict(list)
for line in raw_data:
    sentence = re.sub(r'[^\w\s]|\d','',line['reviewText'].lower())
    words = [word for word in  sentence.split()]
    if len(words) == 0:
        continue
    # Turn the original ID into the index in the dictionary
    UserID = get_key_by_value(line['reviewerID'], id_user_dict)
    ItemID = get_key_by_value(line['asin'], id_item_dict)
    item_real_reviews[ItemID].append(line)
    item_reviews[ItemID].append(review2id(line))
    UserItem = '{}@{}'.format(UserID,ItemID)
    user_item_review[UserItem].append(review2id(line))
    user_item_real_review[UserItem].append(line)
    user_purchased_items[UserID].append(ItemID)
      


# In[6]:

def parseStr(review):
    ItemID = get_key_by_value(review['asin'], id_item_dict)
    rating = review['overall']
    time = review['unixReviewTime']
    review_text = re.sub(r'[^\w\s]|\d','',review['reviewText'].lower())
    text_ids = [str(word_id_dict[word]) for word in review_text.split()]
    if len(text_ids) == 0:
        text_ids =['0','0']
    return '{}||{:1}||'.format(ItemID,rating)+'::'.join(text_ids)+"||{}".format(time)

train = []
validation = []
test = []
for UserID in range(len(id_user_dict)):
    items = user_purchased_items[UserID]
    if UserID % 100 == 0:
        print('Processing {}/{}'.format(UserID,len(id_user_dict)))
    reviews = []
    for ItemID in items:
        UserItem = '{}@{}'.format(UserID,ItemID)
        reviews.append(user_item_real_review[UserItem][0])
    reviews_sorted = sorted(reviews, key=lambda k: k['unixReviewTime']) 
    for i in range(2,len(reviews_sorted)):
        target_review = reviews_sorted[i]
        pre_review = reviews_sorted[i-1]
        pre_pre_review = reviews_sorted[i-2]
        row = '{}&&'.format(UserID)+parseStr(pre_review)+'()'+parseStr(pre_pre_review)+'&&'+parseStr(target_review)
        if i < len(reviews_sorted) - 2:
            train.append(row)
        elif i == len(reviews_sorted) - 2:
            test.append(row)
        elif i == len(reviews_sorted) - 1:
            validation.append(row)


# In[7]:

print('Saving Outputs...')
# Save output

with open('id_user_dict', 'wb') as f:
    pickle.dump(id_user_dict, f)
with open('id_item_dict', 'wb') as f:
    pickle.dump(id_item_dict, f)
with open('id_word_dict', 'wb') as f:
    pickle.dump(id_word_dict, f)
with open('item_real_reviews', 'wb') as f:
    pickle.dump(item_real_reviews, f)
with open('item_reviews', 'wb') as f:
    pickle.dump(item_reviews, f)
with open('user_item_review', 'wb') as f:
    pickle.dump(user_item_review, f)
with open('train_user_purchased_items', 'wb') as f:
    pickle.dump(user_purchased_items, f)    
with open('validation_user_purchased_items', 'wb') as f:
    pickle.dump(user_purchased_items, f)        
with open('test_user_purchased_items', 'wb') as f:
    pickle.dump(user_purchased_items, f)  
with open('train_ided_whole_data', 'w') as out_file:
    out_file.write('\n'.join(train))
with open('validation_ided_whole_data', 'w') as out_file:
    out_file.write('\n'.join(validation))    
with open('test_ided_whole_data', 'w') as out_file:
    out_file.write('\n'.join(test))
with open('train_validation_ided_whole_data', 'w') as out_file:
    out_file.write('\n'.join(train+validation)) 
data_statistics = {
    'max_interaction_length': max_interaction_length,
    'interaction_num': len(raw_data),
    'max_sentence_length': 1,
    'max_sentence_word_length': max_interaction_length,
    'time_bin_number': 1,
    'user_num': len(id_user_dict),
    'item_num': len(id_item_dict),
    'word_num': len(id_word_dict)
}
with open('data_statistics', 'wb') as f:
    pickle.dump(data_statistics, f)


# In[ ]:
model = Word2Vec(full_raw_text, size=64, window=5, min_count=1, workers=4)
word_vectors = model.wv
word_vectors.save_word2vec_format('embeddings.vec')

word2vec2tensor.word2vec2tensor('embeddings.vec', 'tf_embeddings', binary=True)

with open('tf_embeddings_tensor.tsv', 'r') as f:
    embedding = np.loadtxt(f, delimiter='\t')
with open('word_emb.pkl', 'wb') as f:
    pickle.dump(embedding, f)    


print('Done!')

