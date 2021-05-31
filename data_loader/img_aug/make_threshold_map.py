import cv2
import warnings
import pyclipper
import numpy as np
from shapely.geometry import Polygon

warnings.simplefilter("ignore")


class MakeBorderMap(object):
    def __init__(
            self,
            shrink_ratio=0.4,
            thresh_min=0.3,
            thresh_max=0.7
    ):
        """
        :param shrink_ratio: 膨胀比例
        :param thresh_min: 非文字区域threshold_map值
        :param thresh_max: 用于归一化threshold_map  进行一定的缩放，将1缩放到0.7的值，将0缩放到0.3
        D = Area * (1 - r**r) / L
        """
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data):
        img = data['image']
        text_polys = data['polys']
        ignore_tags = data["ignore_tags"]
        canvas = np.zeros(img.shape[:2], dtype=np.float32)
        mask = np.zeros(img.shape[:2], dtype=np.float32)
        for i in range(len(text_polys)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return

        # 计算收缩偏移量D
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(pg) for pg in polygon]
        padding = pyclipper.PyclipperOffset()
        # joinType  当扁平的路径始终无法完美的获取角度信息，他们等价于一系列的圆弧曲线
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # 1. 首先对原始标注框G，采用上述偏移量D来进行扩充，得到的框为Gd
        padded_polygon = np.array(padding.Execute(distance)[0])
        # 使用膨胀的padded_polygon多边形填充mask, 值为1.0
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(
                0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(
                0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            # 2.计算框Gd内所有的点到G的四条边的距离，选择最小的距离（也就是Gd框内像素离它最近的G框的边的距离，下面简称像素到G框的距离）
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            # 3. 将所求的Gd框内所有像素到G框的距离，除以偏移量D进行归一化; 距离限制在[0,1]内, 大于1取值为1, 小于0取值为0
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)

        distance_map = distance_map.min(axis=0)
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        # 使用1减去4中得到的map，这里得到的就是Gd框和Gs框之间的像素到G框最近边的归一化距离; 距离为0,概率为1,距离为1,概率为0;因此1-d
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin: ymax_valid - ymax + height,
                xmin_valid - xmin: xmax_valid - xmax + width
                ],
            canvas[ymin_valid: ymax_valid + 1, xmin_valid: xmax_valid + 1])

    @staticmethod
    def _distance(xs, ys, point_1, point_2):
        """
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        """
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (
                2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        return result
