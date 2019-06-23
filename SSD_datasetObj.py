import os
import numpy as np
from chainercv.datasets import VOCBboxDataset
import chainer
from chainercv.utils import read_image

labels = ('1c', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', '10c', '11c', '12c', '13c', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '10s', '11s', '12s', '13s', '1d', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d', '11d', '12d', '13d', '1h', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', '11h', '12h', '13h')

class CardsDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data_dir, data_range):
        self.data_dir = data_dir #data/created/
        self.label_names = labels #{"1s","2s",...,"13d"}

        self.ids = range(data_range[0],data_range[1])

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        id_ = self.ids[i]
        loaded = np.load("data/created/cards_{:0=8}.npz".format(id_),allow_pickle=True)
        img = loaded["arr_0"]
        label = loaded["arr_1"]
        bbox = loaded["arr_2"]

        bbox = np.array(bbox, dtype=np.float32)
        label = np.array(label, dtype=np.int32)
        return img, bbox, label