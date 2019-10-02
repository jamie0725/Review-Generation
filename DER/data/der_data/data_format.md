As the data is too large, we detail the data format in the following. Anyone who use this code should process the raw data into these formats: 

1. train_ided_whole_data/test_ided_whole_data/validation_ided_whole_data/train_validation_ided_whole_data:
In each line: UserID&&PreviousItemID||rating||WordID::WordID::...::WordID||Time()PreviousItemID||rating||WordID::WordID::...::WordID||Time&&TargetItemID||rating||WordID::WordID::...::WordID||Time
e.g.
593&&602||5.0||1::2025::24::1346::1743::268::102||126()428||5.0||824::211::143::2457::2446::5492::543||848&&726||4.0||0::1889::6946::217::5187::3098::183::788::102::40::2138||838
2. id_user_dict/id_item_dict/id_word_dict
Key: ID ------------ Value: User/Item/Word Name
3. item_reviews
Key: ItemID ------------ Value: All the user reviews received by the Item (processed)
4. item_real_reviews
Key: ItemID ------------ Value: All the user reviews received by the Item (not processed)
5. user_item_review
Key: UserID@ItemID ------------ Value: review from the user to the item
6. train_user_purchased_items/validation_user_purchased_items/test_user_purchased_items
Key: UserID ------------ Value: [PurchasedItemID, PurchasedItemID, ..., PurchasedItemID]
8. data_statistics
Key: max_interaction_length ------------ Value: the max number of interaction length
Key: interaction_num ------------ Value: the number of records
Key: max_sentence_length ------------ Value: the max number of sentences in an item's reviews
Key: max_sentence_word_length ------------ Value: the max number of words in a sentence
Key: time_bin_number ------------ Value: parameter to discrete the continue time
Key: user_num ------------ Value: user number
Key: item_num ------------ Value: item number
Key: word_num ------------ Value: word number

