import os
import cv2
import codecs
import random
import numpy as np
from torch.utils.data import Dataset
from utils.string_utils import CharacterJson
from data_loader.img_aug import *


class RecDataSet(Dataset):
    def __init__(self, config,  logger, mode="train"):
        dataset_config = config[mode]["dataset"]
        global_config = config["global"]
        self.base_dir = dataset_config["data_base_dir"]
        self.mode = mode
        self.logger = logger
        cj = CharacterJson(global_config["character_json_path"])
        self.char2idx = cj.char2idx
        self.max_text_len = global_config["max_text_len"]
        self.data_lines = self.get_image_info_list(dataset_config["ano_file_path"])
        self.transforms = self._transforms_func_lst(dataset_config["transforms"])
        if dataset_config["do_shuffle"]:
            random.shuffle(self.data_lines)

    def __len__(self):
        return len(self.data_lines)

    @staticmethod
    def _transforms_func_lst(config):
        func_lst = []
        for _transform in config:
            operator = list(_transform.keys())[0]
            params = dict() if _transform[operator] is None else _transform[operator]
            func_name = eval(operator)(**params)
            func_lst.append(func_name)
        return func_lst

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

    def rec_label_encoder(self, label_str):
        labels = []
        for char in label_str:
            if char not in self.char2idx.keys():
                continue
            labels.append(self.char2idx[char])
        if len(labels) > self.max_text_len:
            return
        sequence_length = len(labels)
        labels = labels + [self.char2idx["<PAD>"]] * (self.max_text_len - len(labels))
        labels = np.array(labels, dtype=np.int)
        return labels, sequence_length

    def __getitem__(self, index):
        try:
            data_line = self.data_lines[index]
            image_path = os.path.join(self.base_dir, data_line[0])
            label_idx, sequence_length = self.rec_label_encoder(data_line[1])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 默认BGR CHANNEL_LAST
            if image is None:
                self.logger.info(image_path)
            data = {"label_idx": label_idx, "sequence_length": sequence_length, "image": image}
            for _transform in self.transforms:
                data = _transform(data)
        except Exception as e:
            self.logger.error(e)
            data = []

        if not data:
            return self.__getitem__(np.random.randint(self.__len__()))
        return data
