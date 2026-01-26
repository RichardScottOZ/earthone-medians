"""
Test script for earthone-medians serverless compute.

Tests:
1. Submit a small area job
2. Retrieve the GeoTIFF result
3. Verify georeferencing is correct
"""

import os
import tempfile
from earthone_medians import (
    compute_sentinel2_median_serverless,
    retrieve_blob_result,
)


def test_serverless_geotiff():
    """Test serverless compute with GeoTIFF output."""
    
    # Small test area - ~10km x 10km in the Andes, Chile
    bbox = [-70.05, -33.45, -69.95, -33.35]
    start_date = "2023-01-01"
    end_date = "2023-01-31"
    bands = ["B2", "B3", "B4", "B8"]  # Blue, Green, Red, NIR
    
    print("=" * 60)
    print("Testing earthone-medians serverless compute")
    print("=" * 60)
    print(f"BBox: {bbox}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Bands: {bands}")
    print()
    
    # Submit job
    print("Submitting serverless compute job...")
    print("  (This may take a moment to connect to EarthOne...)")
    import sys
    sys.stdout.flush()
    
    result = compute_sentinel2_median_serverless(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=10,
        crs="EPSG:4326",
        cpus=1.0,
        memory=2048,
        max_concurrency=10,
    )
    
    print(f"Job status: {result['status']}")
    
    if result["status"] != "success":
        print(f"ERROR: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"Job ID: {result['job_id']}")
    print(f"Blob ID: {result['blob_id']}")
    print(f"Blob name: {result['blob_name']}")
    print(f"Num images used: {result['num_images']}")
    print(f"Output shape: {result['shape']}")
    print()
    
    # Retrieve and verify GeoTIFF
    print("Retrieving GeoTIFF from blob storage...")
    
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        output_path = tmp.name
    
    retrieve_blob_result(result["blob_id"], output_path=output_path)
    print(f"Saved to: {output_path}")
    
    # Verify georeferencing
    print()
    print("Verifying GeoTIFF georeferencing...")
    import rasterio
    
    with rasterio.open(output_path) as ds:
        print(f"  CRS: {ds.crs}")
        print(f"  Bounds: {ds.bounds}")
        print(f"  Shape: {ds.shape}")
        print(f"  Band count: {ds.count}")
        print(f"  Band descriptions: {ds.descriptions}")
        print(f"  Transform: {ds.transform}")
        
        # Verify bounds match input bbox (approximately)
        minx, miny, maxx, maxy = bbox
        bounds = ds.bounds
        
        tol = 0.01  # Tolerance for floating point comparison
        bounds_ok = (
            abs(bounds.left - minx) < tol and
            abs(bounds.bottom - miny) < tol and
            abs(bounds.right - maxx) < tol and
            abs(bounds.top - maxy) < tol
        )
        
        print()
        if bounds_ok:
            print("✓ Bounds match input bbox")
        else:
            print(f"✗ Bounds mismatch!")
            print(f"  Expected: {bbox}")
            print(f"  Got: [{bounds.left}, {bounds.bottom}, {bounds.right}, {bounds.top}]")
        
        crs_ok = ds.crs.to_string() == "EPSG:4326"
        if crs_ok:
            print("✓ CRS is EPSG:4326")
        else:
            print(f"✗ CRS mismatch: {ds.crs}")
        
        bands_ok = ds.count == len(bands)
        if bands_ok:
            print(f"✓ Band count matches ({ds.count})")
        else:
            print(f"✗ Band count mismatch: expected {len(bands)}, got {ds.count}")
        
        # Read data and check for valid values
        data = ds.read()
        has_data = data.size > 0 and not (data == 0).all()
        if has_data:
            print(f"✓ Contains valid data (min={data.min():.2f}, max={data.max():.2f})")
        else:
            print("✗ No valid data in raster")
    
    # Cleanup
    os.unlink(output_path)
    
    print()
    print("=" * 60)
    all_ok = bounds_ok and crs_ok and bands_ok and has_data
    if all_ok:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_ok


if __name__ == "__main__":
    test_serverless_geotiff()
