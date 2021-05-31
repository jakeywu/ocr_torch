import os
import yaml
import json
import codecs


class ReadConfig(object):
    def __init__(self, yml_path):
        self._yml_path = yml_path
        self.base_conf = self._read_yml_conf()
        self._complement_conf()

    def _read_yml_conf(self):
        with codecs.open(self._yml_path, "r", "utf8") as f:
            conf = yaml.load(f.read(), Loader=yaml.FullLoader)
        return conf

    def _complement_conf(self):
        character_json_path = self.base_conf["global"].get("character_json_path", "")
        if not character_json_path:
            return
        if not os.path.exists(character_json_path):
            raise Exception("path {} not exists".format(character_json_path))

        try:
            with codecs.open(character_json_path, "r", "utf8") as f:
                char2idx = json.loads(f.read())
        except Exception as e:
            raise e
        if "<PAD>" not in char2idx.keys():
            raise Exception("keys <PAD> is not found!")
        if "model_det" in self.base_conf.keys():
            self.base_conf["model_det"]["classes_num"] = len(char2idx)
        if "post_process" in self.base_conf.keys():
            self.base_conf["post_process"]["character_json_path"] = character_json_path
