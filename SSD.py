import numpy as np
from matplotlib import  pyplot as plt
import cv2

import SSD_datasetObj as ssdd
import chainer
from chainercv.links import SSD300
from chainercv.links.model.ssd import multibox_loss

#from chainer.datasets import TransformDataset
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling

import chainermn

import copy
import argparse

from chainercv import transforms
from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation

#chainer.cuda.set_max_workspace_size(1024 * 1024 * 1024)
chainer.config.autotune = True
chainer.config.cv_resize_backend = "cv2"

#データセットを分類
train_dataset = ssdd.CardsDataset("data/created/",(0,128))
valid_dataset = ssdd.CardsDataset("data/created/",(3,4))
test_dataset = ssdd.CardsDataset("data/created/",(1000,1050))

class MultiboxTrainChain(chainer.Chain):
    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def forward(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss

class Transform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        img, bbox, label = in_data

        # 1. Color augmentation
        img = random_distort(img)

        # 2. Random expansion
        if np.random.randint(2):
            img, param = transforms.random_expand(
                img, fill=self.mean, return_param=True)
            bbox = transforms.translate_bbox(
                bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

        # 3. Random cropping
        img, param = random_crop_with_bbox_constraints(
            img, bbox, return_param=True)
        bbox, param = transforms.crop_bbox(
            bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
            allow_outside_center=False, return_param=True)
        label = label[param['index']]

        # 4. Resizing with random interpolatation
        _, H, W = img.shape
        img = resize_with_random_interpolation(img, (self.size, self.size))
        bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

        # 5. Random horizontal flipping
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (self.size, self.size), x_flip=params['x_flip'])

        # Preparation for SSD network
        img -= self.mean
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label

def do():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',choices=('ssd300','ssd512'),default='ssd300')
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--iteration', type=int, default=64)
    parser.add_argument('--step', type=int, nargs='*', default=[8,16])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    model = SSD300(
        n_fg_class=len(ssdd.labels),
        pretrained_model='imagenet'
    )
    model.use_preset('evaluate')
    train_chain = MultiboxTrainChain(model)
    """
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    """
    train = TransformDataset(
        train_dataset,
        Transform(model.coder,model.insize,model.mean),
    )
    train_iter = chainer.iterators.MultiprocessIterator(train,args.batchsize)

    test = test_dataset
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False,shuffle=False)

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))
    
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater,(args.iteration, 'iteration'),args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger= triggers.ManualScheduleTrigger(args.step, 'iteration')
    )
    """
    trainer.extend(
        extensions.Evaluator(
            test_iter, model
        ),
        trigger=triggers.ManualScheduleTrigger(
            args.step + [args.iteration], 'iteration'
        )
    )
    """
    trainer.extend(extensions.ProgressBar(update_interval=1))
    #trainer.extend(extensions.LogReport(trigger=1))
    #trainer.extend(extensions.observe_lr(), trigger=1)
    #trainer.extend(extensions.PrintReport(
    #    ['epoch', 'iteration', 'lr',
    #    'main/loss', 'main/loss/loc', 'main/loss/conf',
    #    'validation/main/map', 'elapsed_time']),
    #    trigger=1)
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'main/loss/loc', 'main/loss/conf'],
                'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['validation/main/map'],
                'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.epoch}.npz'), 
        trigger=(4, 'iteration')
    )

    trainer.run()

if __name__ == '__main__':
    do()
    
"""
model = SSD300(n_fg_class=len(ssdd.labels), pretrained_model='imagenet')
model.use_preset("evaluate")
train_chain = MultiboxTrainChain(model)

train = ssdd.CardsDataset()

print(len(train))
first = train[0]
print(first[0].shape,first[0].dtype)

batchsize = 2
gpu_id = 0
out = 'results'
initial_lr = 0.001
training_epoch = 30
log_interval = 1, 'epoch'
lr_decay_rate = 0.1
lr_decay_timing = [200, 250]

transformed_train_dataset = TransformDataset(train_dataset, ssdd.labels, Transform(model.coder, model.insize, model.mean))

comm = chainermn.create_communicator('pure_nccl')

if comm.rank == 0:
    indices = np.arange(len(transformed_train_dataset))
else:
    indices = None
indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
transformed_train_dataset = transformed_train_dataset.slice[indices]

print("train_dataset:",len(train_dataset))
print("valid_dataset:",len(valid_dataset))

train_iter = chainer.iterators.MultiprocessIterator(transformed_train_dataset, batchsize)
valid_iter = chainer.iterators.SerialIterator(valid_dataset, batchsize, repeat=False, shuffle=False)

optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(train_chain)
for param in train_chain.params():
    if param.name == 'b':
        param.update_rule.add_hook(GradientScaling(2))
    else:
        param.update_rule.add_hook(WeightDecay(0.0005))

updater = training.updaters.StandardUpdater(
    train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(
    updater,
    (training_epoch, 'epoch'), out)

trainer.extend(
    extensions.ExponentialShift('lr', lr_decay_rate, init=initial_lr),
    trigger=triggers.ManualScheduleTrigger(lr_decay_timing, 'epoch'))

trainer.extend(
    DetectionVOCEvaluator(
        valid_iter, model, use_07_metric=False,
        label_names=ssdd.labels),
    trigger=(1, 'epoch'))

trainer.extend(extensions.LogReport(trigger=log_interval))
trainer.extend(extensions.observe_lr(), trigger=log_interval)
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'lr',
     'main/loss', 'main/loss/loc', 'main/loss/conf',
     'validation/main/map', 'elapsed_time']),
    trigger=log_interval)
if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'main/loss/loc', 'main/loss/conf'],
            'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport(
            ['validation/main/map'],
            'epoch', file_name='accuracy.png'))
trainer.extend(extensions.snapshot(
    filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=(10, 'epoch'))

trainer.run()
"""