"""trading_intel.lob — Enhanced LOB reading (13-channel features + CNN).

Modules:
  • features.py — pure-numpy LOB tensor builder with multi-threshold
                  walls, liquidity gaps, depth gradient
  • cnn.py      — Human-style CNN that reads the LOB as a 2D image
"""

from modules.trading_intel.lob.features import (
    N_LOB_CHANNELS_V2,
    CH,
    LOBChannelsV2,
    build_lob_tensor_v2_for_bar,
    stack_bars,
)
from modules.trading_intel.lob.cnn import (
    HumanLOBCNN,
    HumanLOBCNNConfig,
    make_human_lob_cnn,
)

__all__ = [
    "N_LOB_CHANNELS_V2", "CH", "LOBChannelsV2",
    "build_lob_tensor_v2_for_bar", "stack_bars",
    "HumanLOBCNN", "HumanLOBCNNConfig", "make_human_lob_cnn",
]
