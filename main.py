"""
from preprocessing import preprocessing
from create_test_data import create_test_ans_sets
from cnn import create_CNN_model
"""
import Gizou as gz
import numpy as np
from matplotlib import  pyplot as plt
import cv2
import CREATE_TEST_DATASET as ctd
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
"""
input_ans = np.load("data_preprocessed/ans_preped_10_100.npy", allow_pickle=True)
input_img = np.load("data_preprocessed/img_preped_100.npy",allow_pickle=True)

print(input_ans[0].shape)
print(input_img[0].shape)

model = create_CNN_model()

epochs = 5
batch_size = 1
history = model.fit(input_img, input_ans, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(input_img, input_ans, verbose=1)
print()
print("loss:", score[0])
print("accuracy:",score[1])
"""
"""
for i in range(3):
    img,tag = gz.getRandomCard()
    img_shadowed = gz.maskShadow(img)
    img_rotated = gz.imageRotate(img_shadowed,np.random.randint(0,360))
    s,coled = gz.collageNoOverlap(coled,img_rotated,locate)
    if(len(s) != 0):
        print(s)
        locate.append(s)
    else:
        print("ERROR!")
        break
"""


"""
plt.imshow(coled)
plt.show()
"""

ctd.create(0)