"""
EarthOne Medians - Build median time series for Sentinel-2, Landsat, and ASTER.

This package provides tools to compute temporal median composites from satellite
imagery using the EarthOne EarthDaily API serverless compute capability.
"""

__version__ = "0.1.0"
__author__ = "Richard Scott"

from .medians import (
    compute_sentinel2_median,
    compute_landsat_median,
    compute_aster_median,
    compute_median,
)
from .config import (
    SENTINEL2_BANDS,
    LANDSAT_BANDS,
    ASTER_BANDS,
)

__all__ = [
    "compute_sentinel2_median",
    "compute_landsat_median",
    "compute_aster_median",
    "compute_median",
    "SENTINEL2_BANDS",
    "LANDSAT_BANDS",
    "ASTER_BANDS",
]
