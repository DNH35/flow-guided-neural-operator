#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=16 +data=masked_tf_dataset +model=debug_model +data.data=/storage/czw/self_supervised_seeg/all_day_data/manifests ++exp.runner.train_batch_size=64
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import spec_collator

@register_task(name="spec_pretrain")
class SpecPretrain(BaseTask):
    def __init__(self, cfg):
        super(SpecPretrain, self).__init__(cfg)

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss":0, "content_aware_loss":0, "l1_loss":0}
        with torch.no_grad():
            for batch in valid_loader:
                _, valid_outs = criterion(model, batch, device)
                all_outs["loss"] += valid_outs["loss"]
                all_outs["content_aware_loss"] += valid_outs["content_aware_loss"]
                all_outs["l1_loss"] += valid_outs["l1_loss"]
        for key in all_outs:
            all_outs[key] /= len(valid_loader)
        return all_outs

    def build_model(self, cfg):
        assert hasattr(self, "dataset")
        input_dim = self.dataset.get_input_dim()
        # assert input_dim == cfg.input_dim
        return models.build_model(cfg)
 
    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        collate_function_to_use = kwargs.pop('collate_fn', spec_collator)
        
        return data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_function_to_use,
            **kwargs  # Pass the rest of the arguments
        )

