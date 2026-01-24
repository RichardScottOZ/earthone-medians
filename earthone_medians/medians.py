"""
Unified interface for computing medians - combines workbench and serverless approaches.

This module provides backward compatibility and a unified interface.
For new code, prefer using workbench.py or serverless.py directly.
"""

from .workbench import (
    WorkbenchMedianComputer,
    compute_sentinel2_median_workbench,
    compute_landsat_median_workbench,
    compute_aster_median_workbench,
)

from .serverless import (
    ServerlessMedianComputer,
    compute_sentinel2_median_serverless,
    compute_landsat_median_serverless,
    compute_aster_median_serverless,
)

# Backward compatibility - default to workbench for interactive use
MedianComputer = WorkbenchMedianComputer
compute_sentinel2_median = compute_sentinel2_median_workbench
compute_landsat_median = compute_landsat_median_workbench
compute_aster_median = compute_aster_median_workbench

# Also alias the old name
compute_median = WorkbenchMedianComputer

__all__ = [
    "MedianComputer",
    "compute_sentinel2_median",
    "compute_landsat_median",
    "compute_aster_median",
    "compute_median",
    "WorkbenchMedianComputer",
    "ServerlessMedianComputer",
    "compute_sentinel2_median_workbench",
    "compute_landsat_median_workbench",
    "compute_aster_median_workbench",
    "compute_sentinel2_median_serverless",
    "compute_landsat_median_serverless",
    "compute_aster_median_serverless",
]
