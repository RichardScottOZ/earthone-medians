"""Command-line interface for earthone-medians."""

import argparse
import json
import sys
import logging
from typing import List

from .medians import (
    compute_sentinel2_median,
    compute_landsat_median,
    compute_aster_median,
)
from .config import SENTINEL2_BANDS, LANDSAT_BANDS, ASTER_BANDS


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_bbox(bbox_str: str) -> List[float]:
    """Parse bounding box string to list of floats."""
    try:
        bbox = [float(x.strip()) for x in bbox_str.split(",")]
        if len(bbox) != 4:
            raise ValueError("BBox must have 4 values")
        return bbox
    except Exception as e:
        raise ValueError(f"Invalid bbox format: {bbox_str}. Expected: min_lon,min_lat,max_lon,max_lat") from e


def parse_bands(bands_str: str) -> List[str]:
    """Parse bands string to list of band names."""
    if not bands_str:
        return None
    return [b.strip() for b in bands_str.split(",")]


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute temporal median composites from satellite imagery using EarthOne EarthDaily API"
    )
    
    parser.add_argument(
        "sensor",
        choices=["sentinel2", "landsat", "aster"],
        help="Satellite sensor to use"
    )
    
    parser.add_argument(
        "--bbox",
        help="Bounding box as: min_lon,min_lat,max_lon,max_lat"
    )
    
    parser.add_argument(
        "--start-date",
        help="Start date in ISO format (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        help="End date in ISO format (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--bands",
        help="Comma-separated list of bands (e.g., 'B2,B3,B4,B8'). If not specified, uses all science bands."
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        help="Output resolution in meters. If not specified, uses sensor default."
    )
    
    parser.add_argument(
        "--crs",
        help="Output CRS (e.g., 'EPSG:4326', 'EPSG:32633'). Default: EPSG:4326"
    )
    
    parser.add_argument(
        "--api-key",
        help="EarthDaily API key. If not specified, uses EARTHDAILY_API_KEY environment variable."
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save results (JSON format). If not specified, prints to stdout."
    )
    
    parser.add_argument(
        "--list-bands",
        action="store_true",
        help="List available bands for the specified sensor and exit"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # List bands if requested
    if args.list_bands:
        bands_config = {
            "sentinel2": SENTINEL2_BANDS,
            "landsat": LANDSAT_BANDS,
            "aster": ASTER_BANDS
        }
        sensor_bands = bands_config[args.sensor]
        print(f"\nAvailable bands for {args.sensor.upper()}:")
        print("-" * 60)
        for band_id, band_info in sensor_bands.items():
            print(f"{band_id:6s} - {band_info['name']:15s} "
                  f"(Resolution: {band_info['resolution']}m, "
                  f"Wavelength: {band_info['wavelength']})")
        print()
        return 0
    
    # Validate required arguments for computation
    if not args.bbox or not args.start_date or not args.end_date:
        print("Error: --bbox, --start-date, and --end-date are required for median computation", file=sys.stderr)
        parser.print_help()
        return 1
    
    # Parse arguments
    try:
        bbox = parse_bbox(args.bbox)
        bands = parse_bands(args.bands)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Select computation function based on sensor
    compute_functions = {
        "sentinel2": compute_sentinel2_median,
        "landsat": compute_landsat_median,
        "aster": compute_aster_median,
    }
    
    compute_fn = compute_functions[args.sensor]
    
    # Compute median
    try:
        result = compute_fn(
            bbox=bbox,
            start_date=args.start_date,
            end_date=args.end_date,
            bands=bands,
            resolution=args.resolution,
            crs=args.crs,
            api_key=args.api_key,
        )
        
        # Format output
        output_json = json.dumps(result, indent=2)
        
        # Save or print result
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"Results saved to: {args.output}")
        else:
            print(output_json)
        
        return 0
        
    except Exception as e:
        logging.error(f"Error computing median: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
