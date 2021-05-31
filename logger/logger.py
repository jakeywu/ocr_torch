import os
import sys
from logger import log_conf
import logging
import logging.config
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.WARNING)
LOG_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_logger(log_name="data_det", level="INFO", log_path=LOG_PATH):
    """
    :param log_name: 只提供data(debug) 和 mail(Critical)
    :param log_path: 默认目录为sc-log
    :param level: DEBUG, INFO, WARN, ERROR
    :return:
    """
    if log_name == "data_det" and not os.path.isdir(log_path):
        os.makedirs(log_path)
    try:
        logging.config.dictConfig(log_conf.logging_conf(log_path, level))
    except Exception as e1:
        print('日志初始化失败[%s]' % e1)
        sys.exit(1)
    logger = logging.getLogger(log_name)
    return logger
