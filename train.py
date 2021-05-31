import os
import json
import time
import torch
import argparse
from tqdm import tqdm
import torch.distributed
from nets import build_model
from losses import build_loss
from metrics import build_metric
from logger.logger import get_logger
from optimizer import build_optimizer
from config.load_conf import ReadConfig
from postprocess import build_post_process
from data_loader import build_data_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


def main(conf, logger):
    distributed = False
    if not conf["global"]["use_gpu"] or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    model = build_model(conf["model_det"])
    model = model.to(device)

    trainer = Trainer(
        model=model,
        logger=logger,
        conf=conf,
        device=device,
        distributed=distributed
    )

    logger.info("模型初始化完成....")
    time.sleep(2)
    trainer.train()


class Trainer(object):
    def __init__(
            self,
            model,
            logger,
            conf,
            device,
            distributed
    ):
        self._model = model
        self._conf = conf
        self._logger = logger
        self._device = device
        self._global_step = 0
        self._last_epoch = -1
        self._best_epoch = 0
        self._distributed = distributed
        self._global_conf = self._conf["global"]
        self._best_indicator = 0
        self._indicator_name = "best_{}".format(self._conf["metrics"]["main_indicator"])
        self._init_pth_model()
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # rank 标记主机或从机，设置为0表示主机
            # world_size标记使用几个主机，设为1表示1个
            torch.distributed.init_process_group('nccl', init_method="env://", world_size=1, rank=0)
            self._model = torch.nn.parallel.DistributedDataParallel(self._model)
            self.distributed = True

        self._steps_per_epoch = self._get_epoch_data(_len=True)
        self._validate_data = self._get_validate_data()
        self._optimizer, self._schedule = build_optimizer(
            parameters=self._model.parameters(),
            epochs=self._global_conf["epochs"],
            step_each_epoch=self._steps_per_epoch,
            config=self._conf["optimizer"]
        )
        self._criterion = build_loss(self._conf["loss"])
        self._metrics = build_metric(self._conf["metrics"])
        self._post_process = build_post_process(self._conf["post_process"])

        self._start_epoch = 1 if self._last_epoch == -1 else self._last_epoch + 1

    def _get_epoch_data(self, _len=False):
        data = build_data_loader(
            config=self._conf,
            distributed=self._distributed,
            logger=self._logger,
            mode="train"
        )
        if _len:
            return len(data)
        return data

    def _get_validate_data(self):
        data = build_data_loader(
            config=self._conf,
            distributed=self._distributed,
            logger=self._logger,
            mode="validate"
        )
        return data

    def train(self):
        self._model.train()
        self._logger.info("开始训练....")
        time.sleep(1)
        for epoch in range(self._start_epoch, self._global_conf["epochs"] + 1):
            log_start_time = time.time()
            train_loader = self._get_epoch_data()
            for idx, batch in enumerate(train_loader):
                for key, val in batch.items():
                    if not torch.is_tensor(val):
                        continue
                    batch[key] = val.to(self._device)

                self._global_step += 1
                lr = self._optimizer.param_groups[0]["lr"]
                preds = self._model(batch["image"])
                loss_dict = self._criterion(preds, batch)
                self._optimizer.zero_grad()
                loss_dict["loss"].backward()
                self._optimizer.step()
                self._schedule.step()

                indicator_str = ""
                for key, val in loss_dict.items():
                    indicator_str = '{}: {:.4f},'.format(key, val.item())

                if self._global_conf["yml_type"] == "REC":
                    post_result = self._post_process(preds, batch)
                    metrics = self._metrics(post_result)
                    indicator_str += 'acc: {:.4f}, norm_edit_dis: {:.4f},'.format(metrics["acc"],
                                                                                  metrics["norm_edit_dis"])

                if self._global_step % self._global_conf["log_iter"] == 0:
                    batch_time = time.time() - log_start_time
                    info_txt = "【{}/{}】,【{}/{}】, global_step: {}, lr:{:.6}, {} speed: {:.1f} samples/sec"
                    info_txt = info_txt.format(
                        epoch, self._global_conf["epochs"], idx + 1, self._steps_per_epoch, self._global_step, lr,
                        indicator_str, self._global_conf["log_iter"] * preds.size(0) / batch_time,
                    )
                    self._logger.info(info_txt)
                    log_start_time = time.time()

            if epoch % self._global_conf["eval_epoch"] == 0:
                cur_metrics = self._eval()
                self._logger.info(
                    "cur metrics: {}".format(", ".join(["{}:{}".format(k, v) for k, v in cur_metrics.items()])))
                if cur_metrics[self._conf["metrics"]["main_indicator"]] > self._best_indicator:
                    self._best_epoch = epoch
                    self._best_indicator = cur_metrics[self._indicator_name]
                    self._save_pth_model(self._indicator_name, epoch, self._best_epoch, self._best_indicator)

            if epoch % self._global_conf["save_epoch_iter"] == 0:
                file_name = "iter_epoch_{}".format(epoch)
                self._save_pth_model(file_name, epoch, self._best_epoch, self._best_indicator)

            self._save_pth_model("latest", epoch, self._best_epoch, self._best_indicator)

    def _eval(self):
        self._model.eval()
        total_time = 0.0
        with tqdm(total=len(self._validate_data), desc='eval model_det:') as pbar:
            for batch in self._validate_data:
                with torch.no_grad():
                    # 数据进行转换和丢到gpu
                    for key, val in batch.items():
                        if not torch.is_tensor(val):
                            continue
                        batch[key] = val.to(self._device)

                    pbar.update(1)
                    start = time.time()
                    preds = self._model(batch["image"])
                    post_result = self._post_process(preds, batch)
                    total_time += time.time() - start
                    self._metrics(post_result, batch)

        metrics = self._metrics.get_metric()
        self._model.train()
        return metrics

    def _init_pth_model(self):
        init_pth_path = self._global_conf["init_pth_path"]
        if not init_pth_path:
            return self._model
        if not os.path.exists(init_pth_path):
            self._logger.error("pth path {} is not exists".format(init_pth_path))
            raise
        try:
            checkpoint = torch.load(init_pth_path, map_location="cpu")
            self._last_epoch = checkpoint["epoch"]
            self._best_epoch = checkpoint["best_epoch"]
            self._best_indicator = checkpoint[self._indicator_name]

            self._global_step = checkpoint["global_step"]
            self._model.load_state_dict(checkpoint["state_dict"], strict=False)
            self._optimizer.load_state_dict(checkpoint["optimizer"])
            self._schedule.load_state_dict(checkpoint["schedule"])
            for state in self._optimizer.state.values():
                for k, v in state.items():
                    if not torch.is_tensor(v):
                        continue
                    state[k] = v.to(self._device)
        except Exception:
            self._logger.error("model_det init failed")
            raise

    def _save_pth_model(self, file_name, epoch, best_epoch, best_indicator):
        checkpoint = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            self._indicator_name: best_indicator,

            "global_step": self._global_step,
            "state_dict": self._model.module.state_dict() if self._distributed else self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "schedule": self._schedule.state_dict(),
        }
        if not os.path.exists(self._global_conf["save_pth_dir"]):
            os.makedirs(self._global_conf["save_pth_dir"])
        torch.save(checkpoint, os.path.join(self._global_conf["save_pth_dir"], file_name + ".pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="./config/train/det.yml", help="配置文件路径")
    det_conf_path = parser.parse_args().config

    cus_params = ReadConfig(det_conf_path).base_conf
    cus_logger = get_logger(log_path=cus_params["global"]["save_pth_dir"])
    cus_logger.info("相关自定义参数:\n{}".format(json.dumps(cus_params, indent=2, ensure_ascii=False)))
    time.sleep(1)
    main(cus_params, cus_logger)
