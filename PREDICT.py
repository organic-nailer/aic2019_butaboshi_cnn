

from chainercv.links import SSD300

import SSD_datasetObj as ssdd
import chainer
from chainercv.visualizations import vis_bbox
from matplotlib import  pyplot as plt
import numpy as np

# Create a model object
model = SSD300(n_fg_class=len(ssdd.labels), pretrained_model='imagenet')

# Load parameters to the model
chainer.serializers.load_npz(
    'result/snapshot_iter_0.npz', model, path='updater/model:main/model/')

from chainercv import utils

def inference(image_filename):
    # Load a test image
    img = utils.read_image(image_filename, color=True)

    # Perform inference
    bboxes, labels, scores = model.predict([img])

    # Extract the results
    bbox, label, score = bboxes[0], labels[0], scores[0]
    print(bbox)
    print(label)
    print(score)

    # Visualize the detection results
    ax = vis_bbox(img, bbox, label, label_names=ssdd.labels)
    ax.set_axis_off()
    ax.figure.tight_layout()

    plt.show()
    
#inference('hoge.jpg')

def kakunin(i):
    loaded = np.load("data/created/cards_{:0=8}.npz".format(i),allow_pickle=True)
    img = loaded["arr_0"]
    cimg = np.zeros((3,300,300))
    cimg[0] = img[0]
    cimg[1] = img[0]
    cimg[2] = img[0]

    b,l,s = model.predict([cimg])
    print("labels_{}:".format(i),l)

for i in range(0,10):
    kakunin(i)