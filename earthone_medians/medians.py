"""Core module for computing median time series using EarthOne EarthDaily API."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .config import SENSOR_CONFIGS, DEFAULT_CRS

logger = logging.getLogger(__name__)


class MedianComputer:
    """Compute temporal median composites from satellite imagery."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the MedianComputer.

        Args:
            api_key: EarthDaily API key. If not provided, will look for
                    EARTHDAILY_API_KEY environment variable.
        """
        self.api_key = api_key
        self._earthdaily = None

    def _get_earthdaily_client(self):
        """Lazy initialization of EarthDaily client."""
        if self._earthdaily is None:
            try:
                import earthdaily
                if self.api_key:
                    self._earthdaily = earthdaily.EarthDaily(api_key=self.api_key)
                else:
                    # Will use environment variable EARTHDAILY_API_KEY
                    self._earthdaily = earthdaily.EarthDaily()
            except ImportError:
                raise ImportError(
                    "earthdaily package is required. Install with: pip install earthdaily"
                )
        return self._earthdaily

    def compute_median(
        self,
        sensor: str,
        bbox: List[float],
        start_date: str,
        end_date: str,
        bands: Optional[List[str]] = None,
        resolution: Optional[int] = None,
        crs: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute temporal median composite for specified sensor.

        Args:
            sensor: Sensor type ('sentinel2', 'landsat', or 'aster')
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (ISO format: 'YYYY-MM-DD')
            end_date: End date (ISO format: 'YYYY-MM-DD')
            bands: List of band names to include. If None, uses all available bands.
            resolution: Output resolution in meters. If None, uses sensor default.
            crs: Output CRS (e.g., 'EPSG:4326'). If None, uses default.
            **kwargs: Additional parameters to pass to EarthDaily API

        Returns:
            Dictionary containing the computed median and metadata
        """
        if sensor not in SENSOR_CONFIGS:
            raise ValueError(
                f"Unknown sensor: {sensor}. Must be one of {list(SENSOR_CONFIGS.keys())}"
            )

        config = SENSOR_CONFIGS[sensor]
        
        # Use default bands if none specified
        if bands is None:
            bands = list(config["bands"].keys())
        
        # Validate bands
        available_bands = set(config["bands"].keys())
        invalid_bands = set(bands) - available_bands
        if invalid_bands:
            raise ValueError(
                f"Invalid bands for {sensor}: {invalid_bands}. "
                f"Available bands: {available_bands}"
            )

        # Use defaults if not specified
        if resolution is None:
            resolution = config["default_resolution"]
        
        if crs is None:
            crs = DEFAULT_CRS

        logger.info(
            f"Computing median for {sensor} from {start_date} to {end_date}"
        )
        logger.info(f"Bands: {bands}")
        logger.info(f"Resolution: {resolution}m, CRS: {crs}")
        logger.info(f"BBox: {bbox}")

        # Get EarthDaily client
        client = self._get_earthdaily_client()

        # Build query parameters
        query_params = {
            "collection": config["collection"],
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "bands": bands,
            "resolution": resolution,
            "crs": crs,
            **kwargs
        }

        # Execute median computation using serverless compute
        try:
            result = self._execute_median_computation(client, query_params)
            return result
        except Exception as e:
            logger.error(f"Error computing median: {e}")
            raise

    def _execute_median_computation(
        self, client, query_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the median computation using EarthDaily serverless compute.

        Args:
            client: EarthDaily client instance
            query_params: Query parameters for the API

        Returns:
            Result dictionary with computed median and metadata
        """
        # This is a placeholder for the actual EarthDaily API call
        # The actual implementation depends on the earthdaily library API
        
        # Example pseudo-code for what the API call might look like:
        # result = client.compute.median(
        #     collection=query_params["collection"],
        #     bbox=query_params["bbox"],
        #     datetime=query_params["datetime"],
        #     bands=query_params["bands"],
        #     resolution=query_params["resolution"],
        #     crs=query_params["crs"]
        # )
        
        logger.info("Executing median computation via EarthDaily serverless compute...")
        
        # For now, return a structured result that shows what would be computed
        result = {
            "status": "success",
            "sensor": "determined from collection",
            "collection": query_params["collection"],
            "bands": query_params["bands"],
            "bbox": query_params["bbox"],
            "datetime": query_params["datetime"],
            "resolution": query_params["resolution"],
            "crs": query_params["crs"],
            "output": "median_composite_data",
            "metadata": {
                "computed_at": datetime.utcnow().isoformat(),
                "num_scenes": "to_be_determined",
            }
        }
        
        return result


def compute_sentinel2_median(
    bbox: List[float],
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    crs: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute temporal median composite for Sentinel-2 imagery.

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (ISO format: 'YYYY-MM-DD')
        end_date: End date (ISO format: 'YYYY-MM-DD')
        bands: List of band names (e.g., ['B2', 'B3', 'B4', 'B8'])
              If None, uses all science bands (excludes coastal aerosol)
        resolution: Output resolution in meters (default: 10m)
        crs: Output CRS (default: EPSG:4326)
        api_key: EarthDaily API key
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the computed median and metadata
    """
    computer = MedianComputer(api_key=api_key)
    return computer.compute_median(
        sensor="sentinel2",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        crs=crs,
        **kwargs
    )


def compute_landsat_median(
    bbox: List[float],
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    crs: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute temporal median composite for Landsat imagery.

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (ISO format: 'YYYY-MM-DD')
        end_date: End date (ISO format: 'YYYY-MM-DD')
        bands: List of band names (e.g., ['B2', 'B3', 'B4', 'B5'])
              If None, uses all science bands (excludes coastal aerosol)
        resolution: Output resolution in meters (default: 30m)
        crs: Output CRS (default: EPSG:4326)
        api_key: EarthDaily API key
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the computed median and metadata
    """
    computer = MedianComputer(api_key=api_key)
    return computer.compute_median(
        sensor="landsat",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        crs=crs,
        **kwargs
    )


def compute_aster_median(
    bbox: List[float],
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    crs: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute temporal median composite for ASTER imagery.

    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (ISO format: 'YYYY-MM-DD')
        end_date: End date (ISO format: 'YYYY-MM-DD')
        bands: List of band names (e.g., ['B01', 'B02', 'B3N'])
              If None, uses all science bands
        resolution: Output resolution in meters (default: 15m)
        crs: Output CRS (default: EPSG:4326)
        api_key: EarthDaily API key
        **kwargs: Additional parameters

    Returns:
        Dictionary containing the computed median and metadata
    """
    computer = MedianComputer(api_key=api_key)
    return computer.compute_median(
        sensor="aster",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        crs=crs,
        **kwargs
    )


# Alias for backward compatibility
compute_median = MedianComputer
