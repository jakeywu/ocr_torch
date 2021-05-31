from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator='hmean'):
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self._reset()

    def __call__(self, post_result, batch):
        preds = post_result[0]
        gt_polygons, ignore_tags = batch["polys"], batch["ignore_tags"]
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polygons, ignore_tags):
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred]
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """

        metircs = self.evaluator.combine_results(self.results)
        self._reset()
        return metircs

    def _reset(self):
        self.results = []  # clear results
