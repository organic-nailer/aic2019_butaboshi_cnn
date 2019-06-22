import numpy as np
import Gizou as gz


def saveDataset(index,img,names,boxes:"[[left,top,right,bottom],[]]"):
    pack = (img,names,boxes)
    np.save("data/created/cards_{:0=8}.npy".format(index),pack)

def create(index):
    img,name,box = gz.collageRandom()
    saveDataset(index,img,name,box)