#usage
# python3 run_train.py +exp=seeg_wav2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=16 +data=pretrain_wavs_from_disk +model=seeg_wav2vec +data.data=/storage/czw/LanguageEcog/semantics/manifest
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import wav_collator

@register_task(name="seeg_wav_task")
class SeegWavTask(BaseTask):
    def __init__(self, cfg):
        super(SeegWavTask, self).__init__(cfg)

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss":0}
        with torch.no_grad():
            for batch in valid_loader:
                _, valid_outs = criterion(model, batch, device)
                all_outs["loss"] += valid_outs["loss"]
        all_outs["loss"] /= len(valid_loader)
        return all_outs

    def build_model(self, cfg):
        assert hasattr(self, "dataset")
        #input_dim = self.dataset.get_input_dim()
        #assert input_dim == cfg.input_dim
        return models.build_model(cfg)
 
    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, collate_fn=wav_collator, **kwargs)


