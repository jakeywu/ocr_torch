##### 简介
ocr_torch是基于Torch1.8实现的DBNet(2.2M) + CRNN(3.8M)实现的轻量级文字检测识别项目(支持onnx推理). 

##### 项目环境
- linux

- python3.7

- Torch1.8

##### 文本检测模型DBNET

采用mobilenetV3 large作为骨干网络实现

* 训练启动脚本
```
python -m torch.distributed.launch train.py -c config/train/det.yml
```

* 测试启动脚本
```
python predict.py -c config/predict/det.yml
```

##### 文本识别模型CRNN

采用mobilenetV3 small作为骨干网络实现

* 训练启动脚本
```
python -m torch.distributed.launch train.py -c config/train/rec.yml
```

* 测试测试脚本
```
python predict.py -c config/predict/rec.yml
```

##### 文本检测识别合并推理

* 训练推理脚本
```
python lite_ocr.py -c config/lite_ocr.yml
```

##### 主要参考文献及源码
1. DB [https://github.com/MhLiao/DB](https://github.com/MhLiao/DB)
2. PaddleOCR [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
3. DBNET.pytorch [https://github.com/WenmuZhou/DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)
4. Paper [https://arxiv.org/pdf/1911.08947.pdf](https://arxiv.org/pdf/1911.08947.pdf)