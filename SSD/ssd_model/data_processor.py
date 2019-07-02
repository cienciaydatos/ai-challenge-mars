from sklearn.model_selection import train_test_split
from fastai.vision.data import ObjectCategoryList, ObjectItemList, imagenet_stats
from fastai.vision.image import ImageBBox
from pathlib import Path

from fastai.vision.transform import get_transforms
from fastai.vision.data import ObjectItemList, imagenet_stats, bb_pad_collate
from fastai import *
from fastai.vision import *

from ssd_model.data_processor_utils import get_img2bbox

class SSDObjectCategoryList(ObjectCategoryList):
    "`ItemList` for labelled bounding boxes detected using SSD."

    def analyze_pred(self, pred, thresh=0.5, nms_overlap=0.1, ssd=None):
        # def analyze_pred(pred, anchors, grid_sizes, thresh=0.5, nms_overlap=0.1, ssd=None):
        b_clas, b_bb = pred
        a_ic = ssd._actn_to_bb(b_bb, ssd._anchors.cpu(), ssd._grid_sizes.cpu())
        conf_scores, clas_ids = b_clas[:, 1:].max(1)
        conf_scores = b_clas.t().sigmoid()

        out1, bbox_list, class_list = [], [], []

        for cl in range(1, len(conf_scores)):
            c_mask = conf_scores[cl] > thresh
            if c_mask.sum() == 0:
                continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1)
            l_mask = l_mask.expand_as(a_ic)
            boxes = a_ic[l_mask].view(-1, 4)  # boxes are now in range[ 0, 1]
            boxes = (boxes - 0.5) * 2.0  # putting boxes in range[-1, 1]
            ids, count = nms(boxes.data, scores, nms_overlap, 50)  # FIX- NMS overlap hardcoded
            ids = ids[:count]
            out1.append(scores[ids])
            bbox_list.append(boxes.data[ids])
            class_list.append(torch.tensor([cl] * count))

        if len(bbox_list) == 0:
            return None  # torch.Tensor(size=(0,4)), torch.Tensor()

        return torch.cat(bbox_list, dim=0), torch.cat(class_list, dim=0)  # torch.cat(out1, dim=0),

    def reconstruct(self, t, x):
        if t is None: return None
        bboxes, labels = t
        if len((labels - self.pad_idx).nonzero()) == 0: return
        i = (labels - self.pad_idx).nonzero().min()
        bboxes, labels = bboxes[i:], labels[i:]
        return ImageBBox.create(*x.size, bboxes, labels=labels, classes=self.classes, scale=False)


class SSDObjectItemList(ObjectItemList):
    "`ItemList` suitable for object detection."
    _label_cls, _square_show_res = SSDObjectCategoryList, False


class SSDData:
    def __init__(self, path='images/'):
        self.images_path = Path(path)
        bbox_filenames = [filename for filename in os.listdir(path) if filename.endswith('.txt') and filename != 'classes.txt']
        bbox_filenames_train, bbox_filenames_valid = train_test_split(bbox_filenames)
        self.img2bbox = get_img2bbox(bbox_filenames_train)
        self.img2bbox_v = get_img2bbox(bbox_filenames_valid)

    def get_y_func(self, x):
        if x.name in self.img2bbox:
            bboxes, classes = self.img2bbox[x.name]
        else:
            bboxes, classes = self.img2bbox_v[x.name]
        return [bboxes, classes]

    def get_data_generator(self):
        data = (SSDObjectItemList.from_folder(self.images_path)
                .split_by_files(list(self.img2bbox_v.keys()))
                .label_from_func(self.get_y_func)
                .transform(get_transforms(), tfm_y=True, size=224)
                .databunch(bs=64, collate_fn=bb_pad_collate)
                .normalize(imagenet_stats))
        return data

