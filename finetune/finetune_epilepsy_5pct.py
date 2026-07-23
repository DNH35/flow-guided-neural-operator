"""Epilepsy fine-tuning with 5% labeled training data (public-release experiment).

Wrapper around finetune_epilepsy.main using conf/finetune_epilepsy_5pct.yaml.
Does not change the default epilepsy pipeline entrypoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

_FGNO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FGNO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_epilepsy import main


if __name__ == "__main__":
    main(config_path=_FGNO_ROOT / "conf" / "finetune_epilepsy_5pct.yaml")
