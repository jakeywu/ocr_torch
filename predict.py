import os
import cv2
import torch
import json
import argparse
import time
import copy
import codecs
from functools import partial
import numpy as np
import onnx
from config.load_conf import ReadConfig
import onnxruntime as rt
from nets import build_model
from postprocess import build_post_process
from data_loader.img_aug import *


def main(params):
    model = build_model(params["model"])
    post_process = build_post_process(params["post_process"])
    pt = Predictor(model, post_process, params)
    pt.predict()


class Predictor(object):
    def __init__(self, model, post_process, params):
        self._model = model
        self._conf = params["global"]
        self.image_dir_or_path = params["dataset"]["image_dir_or_path"]
        self._transforms = self._transforms_func_lst(params["dataset"]["transforms"])
        self._post_process = post_process
        self._image_list = self._read_images()
        if not os.path.exists(self._conf["res_save_dir"]):
            os.makedirs(self._conf["res_save_dir"])
        if self._conf["use_infer_model"]:
            self.sess = self._convert_train2infer()
        else:
            self.sess = self._init_pth_model()

    @staticmethod
    def _transforms_func_lst(config):
        func_lst = []
        for _transform in config:
            operator = list(_transform.keys())[0]
            params = dict() if _transform[operator] is None else _transform[operator]
            func_name = eval(operator)(**params)
            func_lst.append(func_name)
        return func_lst

    def _convert_train2infer(self):
        if os.path.exists(self._conf["infer_model_path"]):
            return rt.InferenceSession(self._conf["infer_model_path"])

        if not os.path.exists(self._conf["train_model_path"]):
            raise Exception("model_det {} not exists".format(self._conf["train_model_path"]))

        ckpt = torch.load(self._conf["train_model_path"], map_location=torch.device('cpu'))["state_dict"]
        self._model.load_state_dict(ckpt)
        self._model.eval()

        if self._conf["yml_type"] == "DET":
            x = torch.randn(1, 3, 224, 224, requires_grad=True)
            dynamic_axes = {
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"}
            }
        else:
            x = torch.randn(1, 3, 32, 320, requires_grad=True)
            dynamic_axes = {
                "input": {0: "batch_size", 3: "width"},
                "output": {0: "batch_size"}
            }

        torch.onnx.export(
            model=self._model,
            args=x,
            f=self._conf["infer_model_path"],
            export_params=True,
            opset_version=11,
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=["input"],  # 输入名
            output_names=["output"],  # 输出名
            dynamic_axes=dynamic_axes
        )
        try:
            onnx_model = onnx.load(self._conf["infer_model_path"])
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            raise e
        return rt.InferenceSession(self._conf["infer_model_path"])

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

    def _init_pth_model(self):
        if not self._conf["train_model_path"]:
            return self._model
        if not os.path.exists(self._conf["train_model_path"]):
            print("pth path {} is not exists".format(self._conf["train_model_path"]))
            raise
        try:
            checkpoint = torch.load(self._conf["train_model_path"], map_location="cpu")
            self._model.load_state_dict(checkpoint["state_dict"], strict=False)
        except Exception:
            print("model_det init failed")
            raise
        return self._model

    def predict(self):
        self._model.eval()
        result = []
        for image_path in self._image_list:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 默认BGR CHANNEL_LAST
            if image is None:
                print("reading image_path: {} failed".format(image_path))
                continue
            data = {"image": image}
            for _transform in self._transforms:
                data = _transform(data)

            for key, val in data.items():
                data[key] = np.expand_dims(val, axis=0)

            start_time = time.time()
            if self._conf["use_infer_model"]:
                out = self.sess.run(["output"], {"input": data["image"]})[0]
                preds = torch.from_numpy(out)
            else:
                images = torch.from_numpy(data["image"])
                preds = self._model(images)

            print("image: {} \texpend time: {:.4f}".format(image_path, time.time() - start_time))
            post_result = self._post_process(preds, data)
            dt_boxes_json = dict()
            dt_boxes_json["file_name"] = image_path
            if self._conf["yml_type"] == "DET":
                dt_boxes_json["bbox"] = post_result[0][0].tolist()
                dt_boxes_json["score"] = post_result[1][0].tolist()
                self._draw_det_res(image, dt_boxes_json, os.path.basename(image_path))
            else:
                dt_boxes_json["text"] = post_result[0][0]
                dt_boxes_json["score"] = post_result[0][1]
            result.append(dt_boxes_json)

        with codecs.open(os.path.join(self._conf["res_save_dir"], "result.txt"), "a", "utf8") as f:
            for res in result:
                f.write(json.dumps(res, ensure_ascii=False)+"\n")

    def _draw_det_res(self, image, dt_boxes_json, img_name):
        cus_line = partial(cv2.line, color=(255, 255, 0), thickness=1)
        if len(dt_boxes_json) > 0:
            new_im = copy.copy(image)
            for i, box in enumerate(dt_boxes_json["bbox"]):
                score = dt_boxes_json["score"][i]
                cus_line(new_im, (box[0][0], box[0][1]), (box[1][0], box[1][1]))
                cus_line(new_im, (box[1][0], box[1][1]), (box[2][0], box[2][1]))
                cus_line(new_im, (box[2][0], box[2][1]), (box[3][0], box[3][1]))
                cus_line(new_im, (box[3][0], box[3][1]), (box[0][0], box[0][1]))
                cv2.putText(
                    new_im,
                    "{:.3f}".format(score),
                    (box[0][0], box[0][1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(0, 0, 255))

            save_path = os.path.join(self._conf["res_save_dir"], os.path.basename(img_name))
            cv2.imwrite(save_path, new_im)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config/predict/det.yml", help="配置文件路径")
    det_conf_path = parser.parse_args().config

    cus_params = ReadConfig(det_conf_path).base_conf
    print("预测相关参数:\n{}".format(json.dumps(cus_params, indent=2, ensure_ascii=False)))
    main(cus_params)
