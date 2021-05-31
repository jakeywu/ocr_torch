import cv2
import imgaug
import numpy as np
import imgaug.augmenters as iaa


class IaaAugment(object):
    def __init__(
            self,
            flip_prob=0.5,
            affine_rotate=(-5, 5),
            resize_scale=(0.5, 1.5),
    ):
        """
        :param flip_prob: 水平翻转概率
        :param affine_rotate: 放射变换角度
        :param resize_scale: 大小范围
        """
        self.augmenter = iaa.Sequential([
            iaa.Fliplr(p=flip_prob),
            iaa.Affine(rotate=affine_rotate),
            iaa.Resize(resize_scale)
        ])

    def __call__(self, data):
        image = data['image']
        shape = image.shape
        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    @staticmethod
    def may_augment_annotation(aug, data, shape):
        line_polys = []
        for poly in data['polys']:
            keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
            keypoints = aug.augment_keypoints([imgaug.KeypointsOnImage(keypoints, shape=shape)])[0].keypoints
            line_polys.append([(p.x, p.y) for p in keypoints])
        data['polys'] = np.array(line_polys)
        return data


class NormalizeImage(object):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, src_order="hwc", tgt_order="chw"):
        self.tgt_order = tgt_order
        self.src_order = src_order
        shape = (3, 1, 1) if self.src_order == "chw" else (1, 1, 3)
        self.scale = 1.0 / 255.0
        self.mean = np.array(self.MEAN).reshape(shape).astype('float32')
        self.std = np.array(self.STD).reshape(shape).astype('float32')

    def __call__(self, data):
        image = (data["image"].astype('float32') * self.scale - self.mean) / self.std
        if self.src_order == "hwc" and self.tgt_order == "chw":
            data["image"] = image.transpose((2, 0, 1))
        return data


class OutputData(object):
    def __init__(self, keep_keys):
        self.keep_keys = keep_keys

    def __call__(self, data):
        output = dict()
        for key in self.keep_keys:
            output[key] = data[key]
        return output


class ResizeForTest(object):
    def __init__(self, long_size=960):
        self.max_pixes = long_size

    def __call__(self, data):
        image = data["image"]
        src_h, src_w, _ = image.shape

        if src_h > src_w:
            ratio = float(self.max_pixes) / src_h
        else:
            ratio = float(self.max_pixes) / src_w

        resize_h = int(src_h * ratio)
        resize_w = int(src_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        data["image"] = cv2.resize(image, (int(resize_w), int(resize_h)))
        data["src_scale"] = np.array([src_h, src_w], dtype=np.int)
        return data
