# FGNO — Functional Gradient Neural Operator / Flow-Matching Neural Representations

Public release package for FGNO pretraining and fine-tuning across Brain Treebank (BBT), DREAMT, epilepsy, and Sleep-EDF pipelines.

## Layout

```
FGNO/
  conf/                 # Experiment YAML configs (edit paths here)
  models/               # Upstream FFM / MAE backbones + fine-tune heads
  data/                 # Dataset builders (BBT / DREAMT / UniMib) + STFT tensors
  data_preprocess/       # Raw-data -> STFT tensor recipes (epilepsy, sleep_edf, dreamt)
  preprocessors/        # STFT and related front-ends
  tasks/                # Training task registry
  criterions/           # Loss registry
  pretrain/             # Pretraining entrypoints (write to checkpoints/)
  finetune/
    BBT_finetune/       # Speech / pitch / volume / sentence probes
    DREAMT_finetune/    # Wearable PPG / ACC probes
  scripts/              # One-command end-to-end pipeline runners (epilepsy, Sleep-EDF, DREAMT)
  checkpoints/          # Pretrained weights, produced by pretrain/*.py (gitignored)
  outputs/              # Results CSVs / run artifacts (gitignored)
```

## Setup

```bash
cd FGNO
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place pretrained upstream weights under `checkpoints/`, for example:

- `checkpoints/TransformerFFM_big_250_full_sub.pt`

Update data and checkpoint paths in `conf/*.yaml` as needed. Paths may be absolute or relative to the FGNO root.

## Brain Treebank fine-tuning

**Data.** Brain Treebank is ~250GB+ and not included in this repo. Point
`data/braintreebank` at your local copy — either copy it in, or symlink it
(recommended, avoids duplicating the data):

```bash
ln -s /path/to/your/braintreebank_data data/braintreebank
```

It must contain the standard layout: `subject_timings/`, `subject_metadata/`,
`electrode_labels/`, `transcripts/`, and trial HDF5s under `all_subject_data/`.
`conf/custom_{speech,pitch,volume,sentence}_finetune.yaml` each target one
`(subject, electrode, trial)` — edit those fields to point at whichever
subject/trial you have.

**Checkpoint.** All four fine-tuning scripts load an upstream FFM checkpoint
from `model.upstream_ckpt` in their config (default
`checkpoints/TransformerFFM_big_250_full_sub.pt`) — place a pretrained
checkpoint there first (see `pretrain/pretrain_bbt.py` below for the
from-scratch path, or supply one trained elsewhere with matching
`hidden_dim`/`num_heads`/`num_layers`/`feedforward_dim`). Fine-tuning fails
fast with a clear `FileNotFoundError` if it's missing.

From the FGNO root (recommended so imports resolve cleanly):

```bash
# Optional W&B logging
export FGNO_WANDB=1
export FGNO_WANDB_PROJECT=fgno
# export FGNO_WANDB_ENTITY=your-entity   # optional

PYTHONPATH=. python finetune/BBT_finetune/run_finetuning_speech.py
PYTHONPATH=. python finetune/BBT_finetune/run_finetuning_pitch.py
PYTHONPATH=. python finetune/BBT_finetune/run_finetuning_volume.py
PYTHONPATH=. python finetune/BBT_finetune/run_finetuning_sentence.py
```

Or run the helper script:

```bash
bash finetune/BBT_finetune/run_finetuning.sh
```

Each script sweeps electrode × layer × flow-time, trains a linear probe with EMA + early stopping, and writes results under `outputs/`.

Disable W&B with `FGNO_WANDB=0`.

## Chronos baseline (BBT speech)

Reports Chronos embedding probe results across temporal downsample factors (companion baseline to FGNO).

```bash
pip install -r requirements-chronos.txt
PYTHONPATH=. python finetune/BBT_finetune/downsample_chronos_bbt.py
```

Results are written to `outputs/chronos_bbt_downsample_results.csv`. Config: `conf/chronos_bbt_downsample.yaml`.

## Epilepsy / Sleep-EDF: full pipeline (data prep -> pretrain -> finetune -> results)

One command each, from the FGNO root:

```bash
bash scripts/run_epilepsy_pipeline.sh
bash scripts/run_sleepedf_pipeline.sh /path/to/sleep-cassette   # PhysioNet Sleep-EDF raw EDFs
```

Each script skips a stage if its output already exists, so it's safe to
re-run. They cover, in order:

1. **Prep** — turn the raw source data into windowed tensors, then into the
   STFT tensors the models train on. See `data_preprocess/README.md` for what
   each script does and how to run the steps individually. Epilepsy's raw CSV
   is included; Sleep-EDF requires downloading PhysioNet's Sleep-EDF Expanded
   (sleep-cassette) recordings yourself.
2. **Pretrain** — `pretrain/pretrain_epilepsy.py` / `pretrain/pretrain_sleep_edf.py`
   train the FFM backbone and write the checkpoint straight to
   `checkpoints/ffm_<dataset>_pretrain.pt`, where the fine-tuning configs
   expect it.
3. **Finetune** — `finetune/finetune_epilepsy.py` / `finetune/finetune_sleepEDF.py`
   load that checkpoint, sweep encoder layer x flow-time, train a probe per
   combination, and append results to `outputs/<dataset>_clean_*.csv`
   (`conf/finetune_epilepsy.yaml` / `conf/finetune_sleepEDF.yaml`).

Public low-data (5% labeled train split) variants of the fine-tuning stage,
same checkpoint, distinct output CSVs:

```bash
PYTHONPATH=. python finetune/finetune_epilepsy_5pct.py
PYTHONPATH=. python finetune/finetune_sleepEDF_5pct.py
```

Configs: `conf/finetune_epilepsy_5pct.yaml`, `conf/finetune_sleepEDF_5pct.yaml`.

`pretrain_bbt.py` (see "Other entrypoints" below) trains against a
fairseq-style `manifest.tsv` of per-segment `.npy` files under
`data/pretrain_manifests`, built by `data/BBT_data/data/write_pretrain_data_wavs.py`.
That script is not fully wired up for this release: it's a `hydra.main`
entrypoint expecting a `data/BBT_data/conf/` config group that isn't
included, plus a subject/trial split file. **Known gap** — reproducing a BBT
checkpoint from scratch needs that manifest-building step finished first;
fine-tuning against an already-trained checkpoint (above) does not.

## DREAMT (wearable BVP + ACC sleep staging)

**Data.** Raw 64Hz participant CSVs aren't included (14GB, public download
from [PhysioNet DREAMT 2.1.0](https://physionet.org/content/dreamt/2.1.0/)).
There are two parallel pretraining variants sharing the same backbone
(`models/ffm_transformer.py::TransformerModel`) — flow-matching (FFM) and
masked-autoencoding (MAE) — each with its own data prep, since the MAE
windows carry a baked-in mask the FFM ones don't:

```bash
cd FGNO

# Flow-matching (FFM) variant
PYTHONPATH=. python data_preprocess/dreamt/preprocess_dreamt_cross_subject.py --data_dir /path/to/data_64Hz
PYTHONPATH=. python data_preprocess/dreamt/preprocess_dreamt_subject_splits.py --data_dir /path/to/data_64Hz
PYTHONPATH=. python pretrain/pretrain_ffm_dreamt.py
PYTHONPATH=. python finetune/DREAMT_finetune/finetune_dreamt_clean_inference.py  # sleep-stage classification
PYTHONPATH=. python finetune/DREAMT_finetune/finetune_dreamt_BVP_HR.py          # skin-temp regression

# Masked-autoencoder (MAE) variant
PYTHONPATH=. python data_preprocess/dreamt/preprocess_dreamt_masked.py --data_dir /path/to/data_64Hz
PYTHONPATH=. python pretrain/pretrain_mae_dreamt.py
PYTHONPATH=. python finetune/DREAMT_finetune/finetune_dreamt_mae.py            # sleep-stage classification
PYTHONPATH=. python finetune/DREAMT_finetune/finetune_dreamt_mae_skin_temp.py  # skin-temp regression
```

Or, once the data prep and pretraining steps above have produced their
outputs, run all four fine-tuning scripts in one go:

```bash
bash scripts/run_dreamt_finetuning.sh
```

It checks that the required `.pt` splits and checkpoints exist before
starting, and writes results to `outputs/dreamt_{sleep,skin,sleep_mae,skin_mae}.csv`.

All four fine-tuning scripts sweep encoder layer x feature-extraction time,
training a fresh linear probe per combination on top of the frozen upstream
backbone. The two skin-temperature regression scripts
(`finetune_dreamt_BVP_HR.py` FFM-side, `finetune_dreamt_mae_skin_temp.py`
MAE-side) reduce the 20-sample (4Hz) skin-temperature label sequence to its
mean, since the regression head predicts a single scalar per window.

(`preprocess_dreamt_subject_splits.py` writes into the FFM/up-sample data
dir since that's what all four `conf/dreamt_{sleep,skin}{,_mae}_finetune.yaml`
fine-tuning configs expect; see `data_preprocess/dreamt/README.md` for the
full four-script pipeline map.)

See `data_preprocess/dreamt/README.md` for what each script's `.pt` output
actually contains — these are pickled `Dataset`/`DataLoader` objects, not
plain tensors, so every producer/consumer script must run with
`PYTHONPATH=<FGNO root>`.

## Other entrypoints

```bash
PYTHONPATH=. python pretrain/pretrain_bbt.py --data_path conf/custom.yaml
```

## Notes for release users

- BBT configs expect Brain Treebank under `data/braintreebank` with the standard PhysioNet / braintreebank layout (`subject_timings/`, `subject_metadata/`, `transcripts/`, trial HDF5s).
- Fine-tuning scripts share logic in `finetune/BBT_finetune/finetune_utils.py`.
- Chronos is an optional dependency (`requirements-chronos.txt`) and is not required for FGNO FFM fine-tuning.
- W&B logging is on by default everywhere (pretrain and finetune scripts alike) and is safe to disable with `FGNO_WANDB=0` if you don't have a W&B account configured.
- Do not commit raw neural data, W&B runs, or large `.pth` checkpoints.
