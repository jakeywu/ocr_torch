import os
import cv2
import json
import codecs
import random
import numpy as np
from torch.utils.data import Dataset
from data_loader.img_aug import *


class DetDataSet(Dataset):
    def __init__(self, config, logger, mode):
        dataset_conf = config[mode]["dataset"]
        self.base_dir = dataset_conf["data_base_dir"]
        self.mode = mode
        self.logger = logger
        self.data_lines = self.get_image_info_list(dataset_conf["ano_file_path"])
        self._transforms = self._transforms_func_lst(dataset_conf["transforms"])
        if dataset_conf["do_shuffle"]:
            random.shuffle(self.data_lines)

    def __len__(self):
        return len(self.data_lines)

    def get_image_info_list(self, file_path):
        """数据文件以\t分割"""
        lines = []
        with codecs.open(file_path, "r", "utf8") as f:
            for line in f.readlines():
                tmp_data = line.strip().split("\t")
                if len(tmp_data) != 2:
                    self.logger.warn(f"{line}数据格式不对")
                    continue
                image_path = os.path.join(self.base_dir, tmp_data[0])
                if not os.path.exists(image_path):
                    self.logger.warn(f"{image_path}图片文件不存在")
                    continue
                lines.append([tmp_data[0], tmp_data[1]])
        return lines

    @staticmethod
    def det_label_encoder(label_str):
        label = json.loads(label_str)
        boxes = []
        ignore_tags = []
        for bno in range(0, len(label)):
            box = label[bno]["points"]
            txt = label[bno]["transcription"]
            if txt in ["*", "###"]:  # ICDAR为###
                ignore_tags.append(True)
            else:
                ignore_tags.append(False)
            boxes.append(box)

        boxes = np.array(boxes, dtype=np.float)
        ignore_tags = np.array(ignore_tags, dtype=np.bool)
        return boxes, ignore_tags

    @staticmethod
    def _transforms_func_lst(config):
        func_lst = []
        for _transform in config:
            operator = list(_transform.keys())[0]
            params = dict() if _transform[operator] is None else _transform[operator]
            func_name = eval(operator)(**params)
            func_lst.append(func_name)
        return func_lst

    def __getitem__(self, index):
        try:
            data_line = self.data_lines[index]
            image_path = os.path.join(self.base_dir, data_line[0])
            polys, ignore_tags = self.det_label_encoder(data_line[1])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 默认BGR CHANNEL_LAST
            if image is None:
                self.logger.info(image_path)
            data = {"polys": polys, "image": image, "ignore_tags": ignore_tags}
            for _transform in self._transforms:
                data = _transform(data)
        except Exception as e:
            self.logger.error(e)
            data = []

        if not data:
            return self.__getitem__(np.random.randint(self.__len__()))
        return data
