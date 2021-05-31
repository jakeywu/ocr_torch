import os
import cv2
import torch
import json
import argparse
import time
import copy
import codecs
import numpy as np
from functools import partial
from config.load_conf import ReadConfig
import onnxruntime as rt
from postprocess import build_post_process
from data_loader.img_aug import *


def main(params):
    pt = LiteOcr(params)
    pt.predict()


class LiteOcr(object):
    def __init__(self, params):
        self._global_param = params["global"]
        self._det_param = params["det"]
        self._rec_param = params["rec"]
        self.image_dir_or_path = self._global_param["image_dir_or_path"]
        self._image_list = self._read_images()

        self._det_post_process = build_post_process(self._det_param["post_process"])
        rec_conf = self._rec_param["post_process"]
        rec_conf["character_json_path"] = self._global_param["character_json_path"]
        self._rec_post_process = build_post_process(rec_conf)
        self._det_transforms = self._transforms_func_lst(self._det_param["transforms"])

        self.det_sess = rt.InferenceSession(self._global_param["infer_det_path"])
        self.rec_sess = rt.InferenceSession(self._global_param["infer_rec_path"])

        if not os.path.exists(self._global_param["res_save_dir"]):
            os.makedirs(self._global_param["res_save_dir"])

    @staticmethod
    def _transforms_func_lst(config):
        func_lst = []
        for _transform in config:
            operator = list(_transform.keys())[0]
            params = dict() if _transform[operator] is None else _transform[operator]
            func_name = eval(operator)(**params)
            func_lst.append(func_name)
        return func_lst

    def _read_images(self):
        imgs_lists = []
        if self.image_dir_or_path is None or not os.path.exists(self.image_dir_or_path):
            raise Exception("not found any img file in {}".format(self.image_dir_or_path))

        img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff'}
        if os.path.isfile(self.image_dir_or_path) and \
                os.path.splitext(self.image_dir_or_path)[-1][1:].lower() in img_end:
            imgs_lists.append(self.image_dir_or_path)
        elif os.path.isdir(self.image_dir_or_path):
            for single_file in os.listdir(self.image_dir_or_path):
                file_path = os.path.join(self.image_dir_or_path, single_file)
                if os.path.isfile(file_path) and os.path.splitext(file_path)[-1][1:].lower() in img_end:
                    imgs_lists.append(file_path)
        if len(imgs_lists) == 0:
            raise Exception("not found any img file in {}".format(self.image_dir_or_path))
        return imgs_lists

    @staticmethod
    def _get_rotate_crop_image(img, points):
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1]))
        img_crop_height = int(np.linalg.norm(points[0] - points[3]))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])

        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 2:
            dst_img = np.rot90(dst_img)
        return dst_img

    def predict(self):
        result = []
        for image_path in self._image_list:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 默认BGR CHANNEL_LAST
            if image is None:
                print("reading image_path: {} failed".format(image_path))
                continue
            data = {"image": image}
            for _transform in self._det_transforms:
                data = _transform(data)

            for key, val in data.items():
                data[key] = np.expand_dims(val, axis=0)

            start_time = time.time()
            out = self.det_sess.run(["output"], {"input": data["image"]})[0]
            preds = torch.from_numpy(out)

            print("image: {} \texpend time: {:.4f}".format(image_path, time.time() - start_time))
            boxes_batch, scores_batch = self._det_post_process(preds, data)

            results = []
            for idx, (box, score) in enumerate(zip(boxes_batch[0], scores_batch)):
                tmp_box = copy.deepcopy(box)
                tmp_img = self._get_rotate_crop_image(image, tmp_box.astype(np.float32))
                scale = tmp_img.shape[0] * 1.0 / 32
                w = int(tmp_img.shape[1] / scale)
                line_img = RecResizeImg(image_shape=[3, 32, w])({"image": tmp_img})["image"]
                preds = self.rec_sess.run(["output"], {"input": np.expand_dims(line_img, axis=0)})[0]
                line_text, line_score = self._rec_post_process(preds)[0]
                tmp = dict()
                tmp["file_name"] = image_path
                if line_text.strip() != '':
                    tmp["text"] = line_text.replace(" ", "").replace("　", "")
                    bbox = tmp_box.tolist()
                    tmp["score"] = round(float(score*line_score), 3)
                    tmp["bbox"] = [
                        bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1],
                        bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1]
                    ]
                    results.append(tmp)

        with codecs.open(os.path.join(self._global_param["res_save_dir"], "result.txt"), "a", "utf8") as f:
            for res in result:
                f.write(json.dumps(res, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config/lite_ocr.yml", help="配置文件路径")
    det_conf_path = parser.parse_args().config

    cus_params = ReadConfig(det_conf_path).base_conf
    print("预测相关参数:\n{}".format(json.dumps(cus_params, indent=2, ensure_ascii=False)))
    main(cus_params)
