import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


class MakeProbMap(object):
    def __init__(
            self,
            min_text_size=4,
            shrink_ratio=0.4
    ):
        """
        :param min_text_size:　最短边的距离, 根据实际情况决定
        :param shrink_ratio:  收缩比例r
        """
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data):
        image = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = min(np.linalg.norm(polygon[0] - polygon[3]),
                         np.linalg.norm(polygon[1] - polygon[2]))
            width = min(np.linalg.norm(polygon[0] - polygon[1]),
                        np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                subject = [tuple(pg) for pg in polygon]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = []

                # Increase the shrink ratio every time we get multiple polygon returned back
                possible_ratios = np.arange(self.shrink_ratio, 1, self.shrink_ratio)
                np.append(possible_ratios, 1)
                # 这里跟官方DB有一点区别，　但个人认为这个可以更好地检测小文本
                for ratio in possible_ratios:
                    # print(f"Change shrink ratio to {ratio}")
                    distance = polygon_shape.area * (
                        1 - np.power(ratio, 2)) / polygon_shape.length
                    shrinked = padding.Execute(-distance)
                    if shrinked:
                        break

                if not shrinked:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue

                shrink = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt, [shrink.astype(np.int32)], 1)

        data['prob_map'] = gt
        data['prob_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    @staticmethod
    def polygon_area(polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.


if __name__ == "__main__":
    po = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
    print(Polygon(po).area)
    msm = MakeProbMap()
    print(msm.polygon_area(po))
