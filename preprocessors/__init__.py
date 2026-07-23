from .stft import STFTPreprocessor


__all__ = ["STFTPreprocessor",
           "MoreletPreprocessor",
           "SuperletPreprocessor",
           "superlet",
           "WavPreprocessor",
           "SpecPretrained",
           "SpecPooled"
          ]

def build_preprocessor(preprocessor_cfg):
    if preprocessor_cfg.name == "stft":
        extracter = STFTPreprocessor(preprocessor_cfg)
    else:
        raise ValueError("Specify preprocessor")
    return extracter
