"""Volume (RMS) fine-tuning sweep for FGNO on Brain Treebank electrodes."""

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_FGNO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_FGNO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from finetune_utils import ensure_fgno_on_path, run_bbt_finetune_sweep

ensure_fgno_on_path()


def main():
    run_bbt_finetune_sweep(
        config_name="custom_volume_finetune.yaml",
        task_name="volume",
        results_default="outputs/volume_finetuning_result.csv",
        layer_sets=[[i] for i in range(6)],
        epochs=100,
        lr=1e-4,
        early_stopping_patience=20,
    )


if __name__ == "__main__":
    main()
