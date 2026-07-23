# Data preprocessing

Recipes used to produce the tensors under `data/epilepsy_raw`, `data/epilepsy`,
`data/sleepEDF_raw`, and `data/sleepEDF`. The final `*_stft.pt` files are what
`conf/finetune_epilepsy*.yaml` and `conf/finetune_sleepEDF*.yaml` load; the
`*_raw` `.pt` files are the intermediate windowed/labeled tensors before the
STFT front-end.

## Epilepsy (UCI Epileptic Seizure Recognition)

1. `epilepsy/data_preprocess_epilepsy.py` — reads `epilepsy_srcdata.csv`
   (already included), relabels to binary (seizure vs. not), scales, and
   splits 64/16/20 into `data/epilepsy_raw/{train,val,test}.pt`.
2. `epilepsy/preprocess_stft_epilepsy.py` — augments the train split
   (jitter + segment permutation) and STFTs all splits into
   `data/epilepsy/{train,val,test}_stft.pt`.

```bash
cd FGNO
python data_preprocess/epilepsy/data_preprocess_epilepsy.py
python data_preprocess/epilepsy/preprocess_stft_epilepsy.py
```

## Sleep-EDF (PhysioNet Sleep-EDF Expanded, sleep-cassette)

Raw PSG/Hypnogram EDFs are not included (large, public download). Get the
`sleep-cassette` recordings from
[physionet.org/content/sleep-edfx/1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)
first.

1. `sleep_edf/preprocess_sleep_edf.py` — extracts labeled 30s epochs from the
   raw EDFs (`--data_dir` = folder with the downloaded PSG/Hypnogram pairs)
   into per-recording and per-subject `.npz` files. Uses `dhedfreader.py`
   (included).
2. `sleep_edf/generate_train_val_test.py` — combines the per-subject `.npz`
   files into `data/sleepEDF_raw/{train,val,test}.pt` using a fixed 20-subject
   permutation (60/20/20 split).
3. `sleep_edf/preprocess_stft_sleepedf.py` — STFTs the raw tensors into
   `data/sleepEDF/{train,val,test}_stft.pt`.

```bash
cd FGNO
python data_preprocess/sleep_edf/preprocess_sleep_edf.py --data_dir /path/to/sleep-cassette
python data_preprocess/sleep_edf/generate_train_val_test.py
python data_preprocess/sleep_edf/preprocess_stft_sleepedf.py
```

## DREAMT (wearable BVP + ACC sleep staging)

Raw 64Hz participant CSVs are not included (14GB, public download). See
[`data_preprocess/dreamt/README.md`](dreamt/README.md) for the full pipeline
(quality control, cross-subject + single-subject splits, plain and
masked/MAE variants) and exact commands.
