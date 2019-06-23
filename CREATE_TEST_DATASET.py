import numpy as np
import Gizou as gz
import cv2


def saveDataset(index,img,names,boxes:"[[left,top,right,bottom],[]]"):
    gimg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gimg = gimg[:,:,np.newaxis]
    cimg = gimg.transpose((2,0,1))
    cboxes = swapbox(boxes)
    #pack = (cimg,names,boxes)
    np.savez("data/created/cards_{:0=8}.npz".format(index),cimg,names,boxes)

def create(index):
    img,name,box = gz.collageRandom()
    saveDataset(index,img,name,box)

def swapbox(box):
    b = box.transpose((1,0))
    swapped = np.zeros_like(b)
    swapped[0] = b[1]
    swapped[1] = b[0]
    swapped[2] = b[3]
    swapped[3] = b[2]

    return swapped.transpose((1,0))