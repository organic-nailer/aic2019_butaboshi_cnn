import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.datasets.voc import voc_utils
from chainercv.utils import read_image


class VOCBboxDataset(GetterDataset):
    def __init__(self, data_dir='auto', split='train', year='2012',
                 use_difficult=False, return_difficult=False):
        super(VOCBboxDataset, self).__init__()

        if data_dir == 'auto' and year in ['2007', '2012']:
            data_dir = voc_utils.get_voc(year, split)

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label', 'difficult'), self._get_annotations)

        if not return_difficult:
            self.keys = ('img', 'bbox', 'label')

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        id_ = self.ids[i]
        img_path = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_path, color=True)
        return img

    def _get_annotations(self, i):
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = []
        label = []
        difficult = []
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(voc_utils.voc_bbox_label_names.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool)
        return bbox, label, difficult