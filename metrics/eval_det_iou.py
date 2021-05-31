import numpy as np
from shapely.geometry import Polygon


class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):
        def get_union(pd, pg):
            return Polygon(pd).union(Polygon(pg)).area

        def get_intersection_over_union(pd, pg):
            return get_intersection(pd, pg) / get_union(pd, pg)

        def get_intersection(pd, pg):
            return Polygon(pd).intersection(Polygon(pg)).area

        matched_sum = 0
        num_global_care_gt = 0
        num_global_care_det = 0
        det_matched = 0
        iou_mat = np.empty([1, 1])
        gt_pols = []
        det_pols = []
        gt_pol_points = []
        det_pol_points = []
        # Array of Ground Truth Polygons' keys marked as don't Care
        gt_dont_care_pols_num = []
        # Array of Detected Polygons' matched with a don't Care GT
        det_dont_care_pols_num = []

        pairs = []
        det_matched_nums = []
        evaluation_log = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            dont_care = gt[n]['ignore']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gt_pol = points
            gt_pols.append(gt_pol)
            gt_pol_points.append(points)
            if dont_care:
                gt_dont_care_pols_num.append(len(gt_pols) - 1)

        evaluation_log += "GT polygons: " + str(len(gt_pols)) + (
            " (" + str(len(gt_dont_care_pols_num)) + " don't care)\n"
            if len(gt_dont_care_pols_num) > 0 else "\n")

        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            det_pol = points
            det_pols.append(det_pol)
            det_pol_points.append(points)
            if len(gt_dont_care_pols_num) > 0:
                for dont_care_pol in gt_dont_care_pols_num:
                    dont_care_pol = gt_pols[dont_care_pol]
                    intersected_area = get_intersection(dont_care_pol, det_pol)
                    pd_dimensions = Polygon(det_pol).area
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions
                    if precision > self.area_precision_constraint:
                        det_dont_care_pols_num.append(len(det_pols) - 1)
                        break

        evaluation_log += "DET polygons: " + str(len(det_pols)) + (
            " (" + str(len(det_dont_care_pols_num)) + " don't care)\n"
            if len(det_dont_care_pols_num) > 0 else "\n")

        if len(gt_pols) > 0 and len(det_pols) > 0:
            # Calculate IoU and precision matrixs
            output_shape = [len(gt_pols), len(det_pols)]
            iou_mat = np.empty(output_shape)
            gt_rect_mat = np.zeros(len(gt_pols), np.int8)
            det_rect_mat = np.zeros(len(det_pols), np.int8)
            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    p_g = gt_pols[gt_num]
                    p_d = det_pols[det_num]
                    iou_mat[gt_num, det_num] = get_intersection_over_union(p_d, p_g)

            for gt_num in range(len(gt_pols)):
                for det_num in range(len(det_pols)):
                    if gt_rect_mat[gt_num] == 0 and det_rect_mat[det_num] == 0 and\
                            gt_num not in gt_dont_care_pols_num and det_num not in det_dont_care_pols_num:
                        if iou_mat[gt_num, det_num] > self.iou_constraint:
                            gt_rect_mat[gt_num] = 1
                            det_rect_mat[det_num] = 1
                            det_matched += 1
                            pairs.append({'gt': gt_num, 'det': det_num})
                            det_matched_nums.append(det_num)
                            evaluation_log += "Match GT #" + str(gt_num) + " with Det #" + str(det_num) + "\n"

        num_gt_care = (len(gt_pols) - len(gt_dont_care_pols_num))
        num_det_care = (len(det_pols) - len(det_dont_care_pols_num))
        if num_gt_care == 0:
            recall = float(1)
            precision = float(0) if num_det_care > 0 else float(1)
        else:
            recall = float(det_matched) / num_gt_care
            precision = 0 if num_det_care == 0 else float(det_matched) / num_det_care

        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        matched_sum += det_matched
        num_global_care_gt += num_gt_care
        num_global_care_det += num_det_care

        per_sample_metrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'iouMat': [] if len(det_pols) > 100 else iou_mat.tolist(),
            'gtPolPoints': gt_pol_points,
            'detPolPoints': det_pol_points,
            'gtCare': num_gt_care,
            'detCare': num_det_care,
            'gtDontCare': gt_dont_care_pols_num,
            'detDontCare': det_dont_care_pols_num,
            'detMatched': det_matched,
            'evaluationLog': evaluation_log
        }

        return per_sample_metrics

    @staticmethod
    def combine_results(results):
        num_global_care_gt = 0
        num_global_care_det = 0
        matched_sum = 0
        for result in results:
            num_global_care_gt += result['gtCare']
            num_global_care_det += result['detCare']
            matched_sum += result['detMatched']

        method_recall = 0 if num_global_care_gt == 0 else float(
            matched_sum) / num_global_care_gt
        method_precision = 0 if num_global_care_det == 0 else float(
            matched_sum) / num_global_care_det
        if method_recall + method_precision == 0:
            method_hmean = 0
        else:
            method_hmean = 2 * method_recall * method_precision / (method_recall + method_precision)
        method_metrics = {
            'precision': method_precision,
            'recall': method_recall,
            'hmean': method_hmean
        }

        return method_metrics
