import warnings

# Suppress deprecation warnings emitted by optional backends pulled in by audioread
for dep_module in ("aifc", "audioop", "sunau"):
    warnings.filterwarnings(
        "ignore",
        message=f".*'{dep_module}'.*deprecated.*",
        category=DeprecationWarning,
        module="audioread.rawread",
    )

# Librosa warns when n_fft exceeds the short synthetic signals used in tests.
warnings.filterwarnings(
    "ignore",
    message=r"n_fft=.*too large for input signal",
    category=UserWarning,
    module="librosa.core.spectrum",
)
