#usage
#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=False ++exp.runner.num_workers=0 +data=timestamp_data +model=debug_finetune_model ++exp.task.name=debug_finetune_task ++exp.criterion.name=debug_finetune_criterion ++exp.runner.total_steps=1000 ++model.frozen_upstream=True ++exp.runner.checkpoint_step=-1
import logging
import numpy as np
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import finetune_collator
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

@register_task(name="finetune_task")
class FinetuneTask(BaseTask):
    def __init__(self, cfg):
        super(FinetuneTask, self).__init__(cfg)

    def build_model(self, cfg):
        ckpt_path = cfg.upstream_ckpt
        init_state = torch.load(ckpt_path)
        upstream_cfg = init_state["model_cfg"]
        if upstream_cfg.name=='debug_model':
            upstream_cfg.name='masked_tf_model'
        upstream = models.build_model(upstream_cfg)
        assert hasattr(self, "dataset")
        input_dim = self.dataset.get_input_dim()
        assert input_dim == upstream_cfg["input_dim"]
        states = init_state["model"]
        self.load_model_weights(upstream, states, False)
        return models.build_model(cfg, upstream)
        
    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss":0}
        predicts, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch["input"] = batch["input"].to(device)
                _, valid_outs = criterion(model, batch, device, return_predicts=True)

                predicts.append(valid_outs["predicts"])
                labels.append(batch["labels"])
                all_outs["loss"] += valid_outs["loss"]
        labels = np.array([x for y in labels for x in y])
        predicts = [np.array([p]) if len(p.shape)==0 else p for p in predicts]
        predicts = np.concatenate(predicts)
        roc_auc = roc_auc_score(labels, predicts)
        all_outs["loss"] /= len(valid_loader)
        all_outs["roc_auc"] = roc_auc
        return all_outs

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, drop_last=True, collate_fn=finetune_collator, **kwargs)

