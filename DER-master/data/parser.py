import json
import pickle
import numpy as np

from collections import defaultdict

# NOTE: You need at least Python 3.6 to run this code

def string_to_word_ids(id_word_dict, string):
    # Maps a string (sentence) to word IDs as defined in the id_word_dict parameter
    wordids = []
    for word in string.split():
        if word.lower() in id_word_dict:
            wordids.append(str(id_word_dict[word.lower()]))
    return wordids

filename_input = 'reviews_Automotive_5.json'
filename_output = 'train_ided_whole_data'
id_word_dict_path = 'id_word_dict'

# Configurable window length (the paper proposes a value between 2 and 5)
WINDOW_LENGTH = 2

# Load the raw data
raw_data = [json.loads(line) for line in open(filename_input, 'r')]
id_word_dict = pickle.load(open(id_word_dict_path, 'rb'))
word_id_dict = {v: k for k, v in id_word_dict.items()}

# Initialize dictionaries
user_to_reviews = defaultdict(list)
reviews = dict()

for line in raw_data:
    # Map user IDs to review IDs
    user_to_reviews[line['reviewerID']].append(line['asin'])
    # Save reviews by ID for quick access
    reviews[line['asin']] = line

result = []
for user in user_to_reviews:
    # Get all reviews by user
    user_reviews = [reviews[review_id] for review_id in  user_to_reviews[user]]
    for review in user_reviews:
        # Select only reviews that are written before the current one
        prev_reviews = [r for r in user_reviews if r['unixReviewTime'] < review['unixReviewTime']]
        if len(prev_reviews) < WINDOW_LENGTH:
            # Not enough older reviews
            continue
        # Select reviews at random when there are too many. #TODO: this can be improved
        prev_reviews = np.random.choice(prev_reviews, WINDOW_LENGTH)
        # Start formatting the row
        row = f"{user}&&"
        for i, r in enumerate(prev_reviews):
            wordids = string_to_word_ids(word_id_dict, r['reviewText'])
            # Format row
            words = '::'.join(wordids)
            row += f"{r['asin']}||{r['overall']}||{words}||{r['unixReviewTime']}"
            if i < len(prev_reviews)-1:
                # Only added between previous reviews
                row += '()'
        wordids = string_to_word_ids(word_id_dict, r['reviewText'])
        # Finish formatting row and store it      
        words = '::'.join(wordids)
        row += f"&&{review['asin']}||{r['overall']}||{words}||{r['unixReviewTime']}"
        result.append(row)

# Save output
with open(filename_output, 'w') as out_file:
    out_file.write('\n'.join(result))

