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
            api_key: Optional API key parameter (deprecated). 
                    EarthOne authentication uses EARTHONE_CLIENT_ID and 
                    EARTHONE_CLIENT_SECRET environment variables, or 
                    interactive login via `earthone auth login`.
        """
        self.api_key = api_key
        self._earthdaily = None

    def _get_earthdaily_client(self):
        """Lazy initialization of EarthDaily EarthOne client."""
        if self._earthdaily is None:
            try:
                from earthdaily.earthone import Auth
                # Auth will use EARTHONE_CLIENT_ID and EARTHONE_CLIENT_SECRET
                # environment variables, or interactive login if not set
                auth = Auth()
                self._earthdaily = auth
            except ImportError:
                raise ImportError(
                    "earthdaily-earthone package is required. "
                    "Install with: pip install earthdaily-earthone[complete]"
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
        Execute the median computation using EarthOne serverless compute.

        Args:
            client: EarthOne Auth client instance
            query_params: Query parameters for the API

        Returns:
            Result dictionary with computed median and metadata
        """
        try:
            from earthdaily.earthone.catalog import search
            from earthdaily.earthone.dynamic_compute import Mosaic
            from shapely.geometry import box
            
            logger.info("Executing median computation via EarthOne serverless compute...")
            
            # Create bbox geometry
            bbox_geom = box(*query_params["bbox"])
            
            # Parse datetime range
            start_date, end_date = query_params["datetime"].split("/")
            
            # Search catalog for imagery
            logger.info(f"Searching catalog for {query_params['collection']}...")
            search_results = search(
                product_id=query_params["collection"],
                geometry=bbox_geom,
                start_datetime=start_date,
                end_datetime=end_date,
            )
            
            num_scenes = len(list(search_results))
            logger.info(f"Found {num_scenes} scenes")
            
            # Create mosaic with median function
            logger.info("Creating median mosaic...")
            band_str = " ".join(query_params["bands"])
            
            mosaic = Mosaic.from_product_bands(
                query_params["collection"],
                band_str,
                start_datetime=start_date,
                end_datetime=end_date,
                geometry=bbox_geom,
                function="median",
                resolution=query_params["resolution"],
            )
            
            logger.info("Median mosaic created successfully")
            
            result = {
                "status": "success",
                "sensor": query_params["collection"],
                "collection": query_params["collection"],
                "bands": query_params["bands"],
                "bbox": query_params["bbox"],
                "datetime": query_params["datetime"],
                "resolution": query_params["resolution"],
                "crs": query_params["crs"],
                "mosaic": mosaic,
                "metadata": {
                    "computed_at": datetime.utcnow().isoformat(),
                    "num_scenes": num_scenes,
                    "method": "median",
                    "platform": "EarthOne",
                }
            }
            
            return result
            
        except ImportError as e:
            logger.error(f"Missing required EarthOne package: {e}")
            raise ImportError(
                "earthdaily-earthone-dynamic-compute package is required. "
                "Install with: pip install earthdaily-earthone-dynamic-compute"
            )
        except Exception as e:
            logger.error(f"Error during median computation: {e}")
            # Return placeholder result for demonstration/testing
            result = {
                "status": "error",
                "error": str(e),
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
                    "note": "Install earthdaily-earthone[complete] for full functionality",
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
        api_key: Deprecated - use EARTHONE_CLIENT_ID and EARTHONE_CLIENT_SECRET env vars
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
        api_key: Deprecated - use EARTHONE_CLIENT_ID and EARTHONE_CLIENT_SECRET env vars
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
        api_key: Deprecated - use EARTHONE_CLIENT_ID and EARTHONE_CLIENT_SECRET env vars
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
