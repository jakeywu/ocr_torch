import numpy as np
import torch
from utils.string_utils import CharacterJson


class CRnnPostProcess(object):
    def __init__(self, character_json_path):
        cj = CharacterJson(character_json_path)
        self._char2idx = cj.char2idx
        self._idx2char = cj.idx2char

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_token = self._char2idx["<PAD>"]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] == ignored_token:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self._idx2char[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, batch=None):
        if torch.is_tensor(preds):
            preds = preds.detach().numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        preds_idx = preds_idx.transpose(1, 0)
        preds_prob = preds_prob.transpose(1, 0)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if not batch or "label_idx" not in batch.keys():
            return text
        label = batch["label_idx"]
        label = self.decode(label)
        return text, label
