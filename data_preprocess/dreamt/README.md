# DREAMT data preprocessing

Recipes to reproduce `data/dreamt/processed_BVP_ACC_up_sample_datasets/` and
`data/dreamt/processed_BVP_ACC_masked_datasets/`, which
`pretrain/pretrain_ffm_dreamt.py`, `pretrain/pretrain_mae_dreamt.py`, and
`conf/dreamt_*.yaml` load. Ported from a separate research repo's
`bvp_datapreprocessing.ipynb` / `bvp_masked_datapreprocessing.ipynb`.

Raw data is not included (14GB). Download the `data_64Hz` folder from
PhysioNet's [DREAMT 2.1.0](https://physionet.org/content/dreamt/2.1.0/)
(one CSV per participant, e.g. `S002_whole_df.csv`, with `BVP`, `ACC_X/Y/Z`,
`TEMP`, and `Sleep_Stage` columns at 64Hz).

Every `.pt` file below is a **pickled PyTorch `Dataset`/`DataLoader` object**,
not a plain tensor dict like the epilepsy/Sleep-EDF pipelines -- the STFT is
computed lazily in `__getitem__`. Run every script (producer and consumer)
with `PYTHONPATH=<FGNO root>` so `torch.load(...)` can resolve
`data.DREAMT_data.dataset.*` when unpickling.

## Shared code

- `data/DREAMT_data/dataset/preprocessing.py` -- BVP/ACC filtering, artifact
  QC, `STFTPreprocessor` (nperseg=64, noverlap=48, 64Hz -> 33 freq bins x 4
  channels = 132-dim input, 5s/320-sample windows -> 21 time steps).
- `data/DREAMT_data/dataset/pretrain_dataset.py` -- `PretrainDataset`
  (`{'target': ...}`, or `{'masked_input', 'mask_label', 'target'}` when a
  `MaskingConfig` is passed) and the BERT-style block-masking logic.
- `data/DREAMT_data/dataset/finetune_dataset.py` -- `FinetuneDataset`
  (`{'input': ..., 'labels': ...}`).
- `data_preprocess/dreamt/dreamt_dataloaders.py` -- builds the actual
  train/val/test `DataLoader`s from raw CSVs; used by all three scripts below.

## 1. Cross-subject (FFM pretraining + cross-subject finetuning)

```bash
cd FGNO
PYTHONPATH=. python data_preprocess/dreamt/preprocess_dreamt_cross_subject.py --data_dir /path/to/data_64Hz
```

Quality-controls every participant (drops files with no Wake stage or >20%
BVP/ACC artifact rate), splits participants 80/10/20 train/val/test, windows
into 5s epochs, SMOTE-balances the training windows, and writes
`data/dreamt/processed_BVP_ACC_up_sample_datasets/{pretrain,finetune}_{train,val,test}_dataset.pt`.

## 2. Masked (MAE pretraining)

Same pipeline, with each pretrain window additionally carrying a block mask:

```bash
PYTHONPATH=. python data_preprocess/dreamt/preprocess_dreamt_masked.py --data_dir /path/to/data_64Hz
```

Writes `data/dreamt/processed_BVP_ACC_masked_datasets/pretrain_{train,val}_dataset.pt`
(used by `pretrain/pretrain_mae_dreamt.py`).

## 3. Single-subject finetuning splits (sleep-stage classification + skin-temp regression)

```bash
PYTHONPATH=. python data_preprocess/dreamt/preprocess_dreamt_subject_splits.py \
  --data_dir /path/to/data_64Hz --subject S034_whole_df.csv --suffix sub34
```

Writes into `data/dreamt/processed_BVP_ACC_up_sample_datasets/`:
- `finetune_{train,val,test}_dataset_sub34.pt` -- binary Wake-vs-not classification
  for one subject (`conf/dreamt_sleep_finetune.yaml` / `dreamt_sleep_mae_finetune.yaml`).
- `finetune_skin_{train,val,test}_dataset_sub34.pt` -- skin-temperature
  20-sample (4Hz) sequence regression for the same subject
  (`conf/dreamt_skin_finetune.yaml` / `dreamt_skin_mae_finetune.yaml`).

## Fixed vs. the source notebooks

- `get_multichannel_dataloaders`'s cross-subject split was hardcoded to
  `participant_files[:1] / [1:2] / [2:3]` -- i.e. it silently only ever used 3
  participants total, ignoring `split_ratios` and the rest of the dataset.
  That looked like a leftover debug shortcut rather than intended behavior,
  so this version restores a real shuffled split (seed 42) across every
  quality-passing participant.
- `torch.load(...)` needs `weights_only=False` on PyTorch >= 2.6 to unpickle
  these Dataset/DataLoader objects (the default flipped to `True` in 2.6) --
  `pretrain/pretrain_ffm_dreamt.py`, `pretrain/pretrain_mae_dreamt.py`, and
  the `finetune/DREAMT_finetune/*.py` scripts that load these files were
  updated accordingly.
- `pretrain/pretrain_ffm_dreamt.py` had `if __name__ == "main":` (missing
  underscores) -- the training block never ran. Fixed to `"__main__"`.
- `pretrain/pretrain_mae_dreamt.py` imported `FNO`, `GPPrior`,
  `make_grid`/`reshape_for_batchwise` from a `functional_flow_matching`
  package that isn't part of FGNO -- guaranteed `ModuleNotFoundError` on
  import. None of the four were actually called anywhere in the file, so the
  dead imports were removed. `finetune/DREAMT_finetune/finetune_dreamt_clean_inference.py`
  had the same dead import; also removed.
- `data/DREAMT_data/dataset/finetune_dataset.py::FinetuneDataset` previously
  collapsed every label to `np.mean(label_vector)` -- harmless for the
  already-scalar sleep-stage labels, but silently wrong for the 20-sample
  skin-temperature regression sequence (it would have averaged away the
  entire sequence into one number). Fixed to store labels as-is.
- `pretrain/pretrain_mae_dreamt.py` used to define its own full local
  duplicate of the transformer architecture (`ModelConfig`,
  `PositionalEncoding`, `TransformerEncoderInput`, `SpecPredictionHead`,
  `TransformerModel`, `MaskedAutoencoderTransformer`) plus a second,
  independently-written copy of `STFTPreprocessor`/`MaskingConfig`/`mask_inputs`
  -- three of which (`mask_inputs`, `STFTPreprocessor`, and a stray older
  `mask_inputs`) were dead code even in the original, silently shadowed by a
  later redefinition in the same file. It's now rewritten to import the
  shared `models/model_config.py::ModelConfig` and
  `models/dreamt_mae.py::MaskedAutoencoderTransformer` (the same class
  `finetune/DREAMT_finetune/finetune_dreamt_mae*.py` already instantiate to
  *load* this checkpoint) instead of redefining them -- one architecture,
  guaranteed-matching `state_dict` keys on both ends. The one real behavioral
  difference the duplicate had (returning a `(reconstruction, pos_enc)` tuple
  instead of just `reconstruction`) was never actually used by the training
  loop (`pos_enc` was always discarded), so nothing was lost by dropping it.
  Verified end-to-end against real DREAMT data: preprocessed a small subject
  sample, ran a training epoch, and confirmed the resulting checkpoint loads
  cleanly into a freshly-instantiated `MaskedAutoencoderTransformer` -- the
  same load path `finetune_dreamt_mae.py` uses.

## Fine-tuning fixes

All four `finetune/DREAMT_finetune/*.py` scripts (FFM sleep-stage, FFM
skin-temp, MAE sleep-stage, MAE skin-temp) have been run end-to-end against
real DREAMT data and verified working. Along the way:

- `finetune_dreamt_BVP_HR.py` previously loaded `conf/custom_pitch_finetune.yaml`
  -- a Brain Treebank config with no DREAMT-related keys -- and passed the
  raw YAML where a `ModelConfig` was expected. Rewritten to load
  `conf/dreamt_skin_finetune.yaml`, build a proper `ModelConfig`, derive
  `input_dim`/`seq_len` from the data, and reduce the 20-sample skin-temp
  label sequence to its mean (matching the scalar regression head). It runs
  the same skin-temperature regression as `finetune_dreamt_mae_skin_temp.py`,
  against the FFM checkpoint instead of the MAE one.
- `finetune_dreamt_mae_skin_temp.py` had the same label-sequence-vs-scalar
  mismatch; fixed the same way.
- `finetune_dreamt_clean_inference.py` referenced `train_dataset`, a name
  only defined inside the (disabled-by-default) `low_data_mode` branch --
  `NameError` under the shipped config. Fixed to read the sample input from
  `train_loader.dataset`. It also read `cfg_env.train.lr` /
  `cfg_env.train.epochs`, keys that don't exist in
  `conf/dreamt_sleep_finetune.yaml`; hardcoded to match every sibling
  script's convention (`lr=0.0001`, `EPOCHS=100`).
- `models/finetune_classification_dreamt_mae_model.py::FinetuneModel` (used
  by `finetune_dreamt_mae.py`) had no `forward()` method at all --
  `NotImplementedError` on the first call. Added one, mean-pooling the
  hooked encoder-layer outputs (same pattern as the working
  `FinetuneFFMModel`).
- Both DREAMT MAE model classes
  (`finetune_classification_dreamt_mae_model.py`,
  `finetune_regression_dreamt_mae_model.py`) called
  `self.backbone.transformer(inputs, return_intermediate=True)` --
  `TransformerModel.forward`'s actual keyword is `intermediate_rep`, not
  `return_intermediate`. Fixed in both.

See `scripts/run_dreamt_finetuning.sh` for a single command that runs all
four fine-tuning scripts once the data splits and checkpoints below exist.
