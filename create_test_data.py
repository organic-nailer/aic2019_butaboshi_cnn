import numpy as np

numsets = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
marksets = {'c':0,'s':1,'d':2,'h':3}

def create_test_ans(cardset):
    res = np.zeros(52)
    for i in cardset:
      num = i[:-1]
      mark = i[-1]

      res[(numsets[num] - 1) + marksets[mark] * 13] = 1 / len(cardset)
    return np.array([res])

def create_test_ans_sets(cardsets):
    res = np.empty((0,52),float)
    for i in cardsets:
        print(create_test_ans(i).shape)
        res = np.append(res, create_test_ans(i), axis=0)
    return res

#print(create_test_ans({'7c', 'Ks', 'Jc', '6c', '10c', '4s', '9s', '5d', 'Jh', '3d', '4c', '2c', '10d', '10h', '8c', 'Ad', '9h', 'Ac'}))