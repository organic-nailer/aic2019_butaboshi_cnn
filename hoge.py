#import preprocessing
#import create_test_data as ctd
#import cnn

import numpy as np

input_ans,_ = np.load("data/ans_recog_score_10_100.npy",allow_pickle=True)

def create_test_ans_sets(cardsets):
    for i in cardsets:
      print(i)

create_test_ans_sets(input_ans)