import cv2
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

def create_CNN_model():
    
    #入力は720*7200(720^2の画像を横に10個、グレースケールか白黒)
    inputs = Input(shape=(720, 7200, 1))
    
    #一層目
    x = Conv2D(32, kernel_size=(3, 3), strides=(1,1))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = MaxPooling2D()(x)
    
    #二層目
    x = Conv2D(64, kernel_size=(3, 3), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = MaxPooling2D()(x)
    
    #多層ニューラルネットワーク
    x = Flatten()(x) #次元削減->1次元
    x = Dropout(0.4)(x) #ドロップアウト
    x = Dense(128)(x) #全結合ニューラルネットワーク
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(52)(x)
    outputs = Activation("softmax")(x)
    
    model = Model(inputs=inputs, output=outputs)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
    
    return model



print("Hello,world!")