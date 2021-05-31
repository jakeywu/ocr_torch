import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


class DBPostProcess(object):
    def __init__(
            self,
            thresh=0.3,
            box_thresh=0.7,
            max_candidates=1000,
            unclip_ratio=1.6
    ):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, batch):
        """
        prob: text region segmentation map, with shape (N, 1, H, W)
        src_scale: the original shape of images. [[H1, W1], [H2, W2], [H3, W3]...]
        """
        src_scale = batch["src_scale"]
        pred = pred[:, 0, :, :]  # binary
        segmentation = pred > self.thresh
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.size(0)):
            height, width = src_scale[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return np.array(boxes_batch), np.array(scores_batch)

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        """
        pred:  概率图　H * W
        _bitmap: 初始二值图  H * W
        dest_width: 原始宽度
        dest_height: 原始高度
        """
        assert len(_bitmap.shape) == 2
        bitmap = _bitmap.cpu().numpy()
        pred = pred.cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,  # 检测的轮廓不建立等级关系
            cv2.CHAIN_APPROX_SIMPLE  # 压缩方向元素， 只保留该方向的终点坐标
        )
        num_contours = min(len(contours), self.max_candidates)
        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, min_side = self.get_mini_boxes(contour)
            # 返回最小边框的长度
            if min_side < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 2)
            box, min_side = self.get_mini_boxes(box)
            if min_side < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    @staticmethod
    def unclip(box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    @staticmethod
    def get_mini_boxes(contour):
        # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        bounding_box = cv2.minAreaRect(contour)
        #  排序最小外接矩形的4个顶点坐标, 从上往下排序
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
