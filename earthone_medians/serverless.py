"""Serverless compute implementation using EarthOne Compute API Functions."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .config import SENSOR_CONFIGS, DEFAULT_CRS, DEFAULT_MAX_CLOUD_COVER, CLOUD_COVER_PROPERTIES

logger = logging.getLogger(__name__)


class ServerlessMedianComputer:
    """
    Compute temporal median composites using EarthOne Compute API Functions.
    
    This approach submits batch jobs to EarthOne's serverless infrastructure,
    ideal for large-scale processing.
    """

    def __init__(self):
        """Initialize the ServerlessMedianComputer."""
        self._auth = None

    def _get_auth(self):
        """Lazy initialization of EarthOne Auth."""
        if self._auth is None:
            try:
                from earthdaily.earthone.auth import Auth
                self._auth = Auth()
            except ImportError:
                raise ImportError(
                    "earthdaily-earthone package is required. "
                    "Install with: pip install earthdaily-earthone[complete]"
                )
        return self._auth

    def compute_median(
        self,
        sensor: str,
        bbox: List[float],
        start_date: str,
        end_date: str,
        bands: Optional[List[str]] = None,
        resolution: Optional[int] = None,
        crs: Optional[str] = None,
        cpus: float = 1.0,
        memory: int = 2048,
        max_cloud_cover: Optional[float] = None,
        max_concurrency: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute temporal median composite using serverless batch processing.

        Submits a job to EarthOne's serverless compute infrastructure with
        configurable resources. 
        
        PLATFORM LIMITS:
        - Maximum 1,000 concurrent jobs per user (platform-wide)
        - Function-specific concurrency limit: 10 (set in Function creation)
        - Compute seconds quota: varies by account (monthly reset)
        - See SERVERLESS_LIMITS.md for detailed quota information

        Args:
            sensor: Sensor type ('sentinel2', 'landsat', or 'aster')
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date (ISO format: 'YYYY-MM-DD')
            end_date: End date (ISO format: 'YYYY-MM-DD')
            bands: List of band names to include
            resolution: Output resolution in meters
            crs: Output CRS
            cpus: CPU allocation for the compute job (default: 1.0, range: 0.25-8+)
            memory: Memory allocation in MB (default: 2048, range: 512-32768+)
            max_cloud_cover: Maximum cloud cover percentage (0-100). Default: 20%.
                           Filters imagery to only include scenes with cloud cover
                           less than or equal to this threshold.
            max_concurrency: Maximum concurrent invocations of this function (default: 10).
                           Platform limit is 1,000 concurrent jobs per user.
            **kwargs: Additional parameters

        Returns:
            Dictionary containing job information and results with keys:
            - job_id: Unique job identifier
            - status: Job completion status
            - function_name: Name of the compute function
            - result: Computation results
            - logs: Job execution logs
        
        Raises:
            ValueError: If sensor is unknown or bands are invalid
            ImportError: If required packages are not installed
            Exception: If rate limit exceeded or job submission fails
        """
        if sensor not in SENSOR_CONFIGS:
            raise ValueError(
                f"Unknown sensor: {sensor}. Must be one of {list(SENSOR_CONFIGS.keys())}"
            )

        config = SENSOR_CONFIGS[sensor]
        
        if bands is None:
            bands = list(config["bands"].keys())
        
        if resolution is None:
            resolution = config["default_resolution"]
        
        if crs is None:
            crs = DEFAULT_CRS
        
        if max_cloud_cover is None:
            max_cloud_cover = DEFAULT_MAX_CLOUD_COVER

        logger.info(f"Submitting serverless compute job for {sensor}")
        logger.info(f"Bands: {bands}, Resolution: {resolution}m, CRS: {crs}")
        logger.info(f"Max cloud cover: {max_cloud_cover}%")

        # Get auth
        self._get_auth()

        try:
            from earthdaily.earthone.compute import Function
            from earthdaily.earthone.catalog import search
            import numpy as np
            
            # Define the median computation function
            def compute_median_job(collection, bbox, start_date, end_date, bands, resolution, crs, max_cloud_cover, cloud_cover_property, job_name):
                """Function that runs on EarthOne compute infrastructure.
                
                Saves result to EarthOne Storage as a GeoTIFF blob, returns reference.
                """
                import numpy as np
                import io
                from datetime import datetime
                from earthdaily.earthone.catalog import search, Blob
                from earthdaily.earthone.compute import Result
                from earthdaily.earthone import raster
                from shapely.geometry import box
                import rasterio
                from rasterio.transform import from_bounds
                
                # Create bbox geometry
                bbox_geom = box(*bbox)
                minx, miny, maxx, maxy = bbox
                
                # Build property filter for cloud cover
                property_filter = {
                    cloud_cover_property: {"lte": max_cloud_cover}
                }
                
                # Search for imagery with cloud cover filter
                results = search(
                    product_id=collection,
                    geometry=bbox_geom,
                    start_datetime=start_date,
                    end_datetime=end_date,
                    property_filter=property_filter,
                )
                
                images = list(results)
                num_images = len(images)
                
                if num_images == 0:
                    return {"error": "No images found", "num_images": 0}
                
                # Load and stack rasters
                arrays = []
                for img in images:
                    arr = raster.ndarray(
                        img.id,
                        bands=bands,
                        resolution=resolution,
                        crs=crs,
                        bounds=bbox_geom.bounds
                    )
                    arrays.append(arr)
                
                # Stack and compute median
                stack = np.stack(arrays, axis=0)
                median_result = np.median(stack, axis=0)
                
                # Save as GeoTIFF
                num_bands = median_result.shape[0] if median_result.ndim == 3 else 1
                height = median_result.shape[-2]
                width = median_result.shape[-1]
                transform = from_bounds(minx, miny, maxx, maxy, width, height)
                
                buffer = io.BytesIO()
                with rasterio.open(
                    buffer, 'w', driver='GTiff',
                    height=height, width=width, count=num_bands,
                    dtype=median_result.dtype, crs=crs, transform=transform,
                    compress='lzw',
                ) as dst:
                    if median_result.ndim == 3:
                        for i in range(num_bands):
                            dst.write(median_result[i], i + 1)
                            dst.set_band_description(i + 1, bands[i] if i < len(bands) else f"band_{i+1}")
                    else:
                        dst.write(median_result, 1)
                
                buffer.seek(0)
                
                # Build descriptive blob name
                blob_name = f"median_{collection.split(':')[-1]}_{minx}_{miny}_{maxx}_{maxy}_{start_date}_{end_date}.tif"
                
                blob = Blob(
                    name=blob_name,
                    data=buffer.getvalue(),
                    attributes={
                        "type": "median_composite",
                        "format": "GeoTIFF",
                        "collection": collection,
                        "bands": bands,
                        "bbox": bbox,
                        "start_date": start_date,
                        "end_date": end_date,
                        "resolution": resolution,
                        "crs": crs,
                        "num_images": num_images,
                        "shape": list(median_result.shape),
                        "dtype": str(median_result.dtype),
                        "created_at": datetime.utcnow().isoformat(),
                    }
                )
                blob.save()
                
                # Return reference to stored blob (small payload)
                return Result(
                    data={
                        "success": True,
                        "blob_id": blob.id,
                        "blob_name": blob.name,
                        "num_images": num_images,
                        "shape": list(median_result.shape),
                        "dtype": str(median_result.dtype),
                    },
                    attributes={
                        "job_name": job_name,
                        "collection": collection,
                    }
                )
            
            # Create and register the Function
            logger.info("Creating serverless compute function...")
            job_name = f"earthone-medians-{sensor}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            func = Function(
                compute_median_job,
                name=job_name,
                image="python3.10:latest",
                cpus=cpus,
                memory=memory,
                timeout=3600,  # 1 hour timeout
                maximum_concurrency=max_concurrency,
                requirements=[
                    "earthdaily-earthone>=5.0.0",
                    "numpy>=2.0.0",
                    "shapely>=2.0.0",
                    "rasterio>=1.3.0",
                ]
            )
            
            # Save the function
            logger.info("Registering function with EarthOne...")
            func.save()
            
            # Submit the job
            logger.info("Submitting compute job...")
            job = func(
                config["collection"],
                bbox,
                start_date,
                end_date,
                bands,
                resolution,
                crs,
                max_cloud_cover,
                CLOUD_COVER_PROPERTIES[sensor],
                job_name,  # Pass job name for blob naming
            )
            
            logger.info(f"Job submitted: {job.id}")
            logger.info("Waiting for job completion...")
            
            # Wait for completion (with timeout)
            job.wait_for_completion(timeout=3600)
            
            # Get results
            result = job.result()
            logs = job.log()
            
            logger.info("Job completed successfully")
            
            return {
                "status": "success",
                "method": "serverless_compute",
                "sensor": sensor,
                "collection": config["collection"],
                "bands": bands,
                "bbox": bbox,
                "datetime": f"{start_date}/{end_date}",
                "resolution": resolution,
                "crs": crs,
                "job_id": job.id,
                "function_name": func.name,
                "blob_id": result.get("blob_id") if isinstance(result, dict) else None,
                "blob_name": result.get("blob_name") if isinstance(result, dict) else None,
                "num_images": result.get("num_images") if isinstance(result, dict) else None,
                "shape": result.get("shape") if isinstance(result, dict) else None,
                "logs": logs,
                "metadata": {
                    "computed_at": datetime.utcnow().isoformat(),
                    "cpus": cpus,
                    "memory_mb": memory,
                    "platform": "EarthOne Compute API",
                    "storage": "EarthOne Blob Storage",
                }
            }
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            raise ImportError(
                "earthdaily-earthone[complete] package is required. "
                "Install with: pip install earthdaily-earthone[complete]"
            )
        except Exception as e:
            logger.error(f"Error during serverless compute: {e}")
            return {
                "status": "error",
                "error": str(e),
                "sensor": sensor,
                "collection": config["collection"],
                "bands": bands,
                "bbox": bbox,
                "metadata": {
                    "computed_at": datetime.utcnow().isoformat(),
                    "note": "Serverless compute job failed",
                }
            }


def compute_sentinel2_median_serverless(
    bbox: List[float],
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    crs: Optional[str] = None,
    cpus: float = 1.0,
    memory: int = 2048,
    max_cloud_cover: Optional[float] = None,
    max_concurrency: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Sentinel-2 median using serverless batch processing.
    
    Submits a job to EarthOne Compute API for scalable processing.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (ISO format: 'YYYY-MM-DD')
        end_date: End date (ISO format: 'YYYY-MM-DD')
        bands: List of band names
        resolution: Output resolution in meters
        crs: Output CRS
        cpus: CPU allocation (default: 1.0)
        memory: Memory in MB (default: 2048)
        max_cloud_cover: Maximum cloud cover percentage (0-100, default: 20)
        max_concurrency: Max parallel invocations of this function (default: 10)
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with job information and results
    """
    computer = ServerlessMedianComputer()
    return computer.compute_median(
        sensor="sentinel2",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        crs=crs,
        cpus=cpus,
        memory=memory,
        max_cloud_cover=max_cloud_cover,
        max_concurrency=max_concurrency,
        **kwargs
    )


def compute_landsat_median_serverless(
    bbox: List[float],
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    crs: Optional[str] = None,
    cpus: float = 1.0,
    memory: int = 2048,
    max_cloud_cover: Optional[float] = None,
    max_concurrency: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute Landsat median using serverless batch processing.
    
    Submits a job to EarthOne Compute API for scalable processing.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (ISO format: 'YYYY-MM-DD')
        end_date: End date (ISO format: 'YYYY-MM-DD')
        bands: List of band names
        resolution: Output resolution in meters
        crs: Output CRS
        cpus: CPU allocation (default: 1.0)
        memory: Memory in MB (default: 2048)
        max_cloud_cover: Maximum cloud cover percentage (0-100, default: 20)
        max_concurrency: Max parallel invocations of this function (default: 10)
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with job information and results
    """
    computer = ServerlessMedianComputer()
    return computer.compute_median(
        sensor="landsat",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        crs=crs,
        cpus=cpus,
        memory=memory,
        max_cloud_cover=max_cloud_cover,
        max_concurrency=max_concurrency,
        **kwargs
    )


def compute_aster_median_serverless(
    bbox: List[float],
    start_date: str,
    end_date: str,
    bands: Optional[List[str]] = None,
    resolution: Optional[int] = None,
    crs: Optional[str] = None,
    cpus: float = 1.0,
    memory: int = 2048,
    max_cloud_cover: Optional[float] = None,
    max_concurrency: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute ASTER median using serverless batch processing.
    
    Submits a job to EarthOne Compute API for scalable processing.
    
    Args:
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        start_date: Start date (ISO format: 'YYYY-MM-DD')
        end_date: End date (ISO format: 'YYYY-MM-DD')
        bands: List of band names
        resolution: Output resolution in meters
        crs: Output CRS
        cpus: CPU allocation (default: 1.0)
        memory: Memory in MB (default: 2048)
        max_cloud_cover: Maximum cloud cover percentage (0-100, default: 20)
        max_concurrency: Max parallel invocations of this function (default: 10)
        **kwargs: Additional parameters
    
    Returns:
        Dictionary with job information and results
    """
    computer = ServerlessMedianComputer()
    return computer.compute_median(
        sensor="aster",
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        crs=crs,
        cpus=cpus,
        memory=memory,
        max_cloud_cover=max_cloud_cover,
        max_concurrency=max_concurrency,
        **kwargs
    )


def retrieve_blob_result(blob_id: str, output_path: Optional[str] = None):
    """
    Retrieve a median composite GeoTIFF from EarthOne Blob storage.
    
    Args:
        blob_id: The blob ID returned from a serverless compute job
        output_path: Optional path to save the GeoTIFF locally
        
    Returns:
        If output_path provided: saves file and returns path
        Otherwise: returns rasterio DatasetReader for in-memory access
    """
    import io
    from earthdaily.earthone.catalog import Blob
    import rasterio
    
    blob = Blob.get(blob_id)
    data = blob.data
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(data)
        return output_path
    
    buffer = io.BytesIO(data)
    return rasterio.open(buffer)
