"""trading_intel.training — end-to-end training scripts.

This is intentionally script-heavy (not library-heavy). Each script
is a single-file pipeline that can be invoked as:

    python -m modules.trading_intel.training.train_hybrid \\
        --features ... --ssl-embeddings ... --output ...
"""
