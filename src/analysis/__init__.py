"""Analysis module for RAPID results processing and visualization."""

import os
from pathlib import Path

def _ensure_mplconfig() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path("analysis_outputs") / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)

_ensure_mplconfig()

from .results_manager import ResultsManager
from .config import AnalysisConfig

# Import visualization components if available
try:
    from .visualization import VisualizationManager, HeatmapPlotter, MetricsPlotter
    visualization_available = True
except ImportError:
    visualization_available = False

# Import multi-replica analysis components if available
try:
    from .multi_replica import MultiReplicaAnalyzer, MultiReplicaPlotter, AggregatedMetrics
    multi_replica_available = True
except ImportError:
    multi_replica_available = False

# Build __all__ list based on available components
__all__ = ["ResultsManager", "AnalysisConfig"]

if visualization_available:
    __all__.extend([
        "VisualizationManager",
        "HeatmapPlotter", 
        "MetricsPlotter",
    ])

if multi_replica_available:
    __all__.extend([
        "MultiReplicaAnalyzer",
        "MultiReplicaPlotter", 
        "AggregatedMetrics"
    ]) 
