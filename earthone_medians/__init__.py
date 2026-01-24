"""
EarthOne Medians - Build median time series for Sentinel-2, Landsat, and ASTER.

This package provides TWO approaches to compute temporal median composites:

1. **Workbench/Interactive** (workbench.py):
   - Uses earthdaily.earthone.dynamic_compute.Mosaic
   - Ideal for interactive analysis in Jupyter notebooks
   - Fast visualization and exploration
   - Use: compute_*_median_workbench() functions

2. **Serverless Compute** (serverless.py):
   - Uses earthdaily.earthone.compute.Function
   - Ideal for large-scale batch processing
   - Scalable cloud infrastructure
   - Use: compute_*_median_serverless() functions
"""

__version__ = "0.1.0"
__author__ = "Richard Scott"

# Import workbench/interactive functions
from .workbench import (
    WorkbenchMedianComputer,
    compute_sentinel2_median_workbench,
    compute_landsat_median_workbench,
    compute_aster_median_workbench,
)

# Import serverless compute functions
from .serverless import (
    ServerlessMedianComputer,
    compute_sentinel2_median_serverless,
    compute_landsat_median_serverless,
    compute_aster_median_serverless,
)

# Import configuration
from .config import (
    SENTINEL2_BANDS,
    LANDSAT_BANDS,
    ASTER_BANDS,
)

# Backward compatibility - default to workbench approach
compute_sentinel2_median = compute_sentinel2_median_workbench
compute_landsat_median = compute_landsat_median_workbench
compute_aster_median = compute_aster_median_workbench
MedianComputer = WorkbenchMedianComputer

__all__ = [
    # Workbench/Interactive API
    "WorkbenchMedianComputer",
    "compute_sentinel2_median_workbench",
    "compute_landsat_median_workbench",
    "compute_aster_median_workbench",
    # Serverless Compute API
    "ServerlessMedianComputer",
    "compute_sentinel2_median_serverless",
    "compute_landsat_median_serverless",
    "compute_aster_median_serverless",
    # Backward compatibility (defaults to workbench)
    "MedianComputer",
    "compute_sentinel2_median",
    "compute_landsat_median",
    "compute_aster_median",
    # Configuration
    "SENTINEL2_BANDS",
    "LANDSAT_BANDS",
    "ASTER_BANDS",
]
