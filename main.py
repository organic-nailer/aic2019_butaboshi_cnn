from preprocessing import preprocessing
from create_test_data import create_test_ans_sets
from matplotlib import  pyplot as plt
from cnn import create_CNN_model

import numpy as np
"""
input_ans,_ = np.load("data/ans_recog_score_10_100.npy",allow_pickle=True)

output_ans = create_test_ans_sets(input_ans)

print(output_ans)

np.save("data_preprocessed/ans_preped_10_100.npy",output_ans,allow_pickle=True)
"""
"""
input_img = np.load("data/intput_data_100.npy",allow_pickle=True)

output_img = preprocessing(input_img)

plt.imshow(output_img[0,:,:,0])
plt.gray()
plt.show()

np.save("data_preprocessed/img_preped_100.npy",output_img,allow_pickle=True)
"""

input_ans = np.load("data_preprocessed/ans_preped_10_100.npy", allow_pickle=True)
input_img = np.load("data_preprocessed/img_preped_100.npy",allow_pickle=True)

print(input_ans[0].shape)
print(input_img[0].shape)

model = create_CNN_model()

epochs = 5
batch_size = 1
history = model.fit(input_img, input_ans, batch_size=batch_size, epochs=epochs, verbose=1)