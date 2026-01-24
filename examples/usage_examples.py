"""Example usage of earthone-medians library."""

from earthone_medians import (
    compute_sentinel2_median,
    compute_landsat_median,
    compute_aster_median,
    MedianComputer,
)


def example_sentinel2():
    """Example: Compute Sentinel-2 median for a region."""
    print("Example: Sentinel-2 Median")
    print("-" * 50)
    
    # Define area of interest (bounding box)
    bbox = [115.0, -32.0, 116.0, -31.0]  # Perth, Australia region
    
    try:
        # Compute median for 2023
        result = compute_sentinel2_median(
            bbox=bbox,
            start_date="2023-01-01",
            end_date="2023-12-31",
            bands=["B2", "B3", "B4", "B8"],  # Blue, Green, Red, NIR
            resolution=10,  # 10m resolution
            crs="EPSG:4326",
        )
        
        print(f"Status: {result['status']}")
        print(f"Bands: {result['bands']}")
        print(f"Resolution: {result['resolution']}m")
    except ImportError as e:
        print(f"Note: {e}")
        print("This is a demonstration of the API structure.")
    print()


def example_landsat():
    """Example: Compute Landsat median for a region."""
    print("Example: Landsat Median")
    print("-" * 50)
    
    # Define area of interest
    bbox = [-122.5, 37.5, -122.0, 38.0]  # San Francisco region
    
    try:
        # Compute median with all available science bands
        result = compute_landsat_median(
            bbox=bbox,
            start_date="2022-01-01",
            end_date="2022-12-31",
            resolution=30,  # 30m resolution (Landsat default)
            crs="EPSG:32610",  # UTM Zone 10N
        )
        
        print(f"Status: {result['status']}")
        print(f"Bands: {result['bands']}")
        print(f"Resolution: {result['resolution']}m")
    except ImportError as e:
        print(f"Note: {e}")
        print("This is a demonstration of the API structure.")
    print()


def example_aster():
    """Example: Compute ASTER median for a region."""
    print("Example: ASTER Median")
    print("-" * 50)
    
    # Define area of interest
    bbox = [138.0, -35.0, 139.0, -34.0]  # Adelaide, Australia region
    
    try:
        # Compute median with selected bands
        result = compute_aster_median(
            bbox=bbox,
            start_date="2020-01-01",
            end_date="2023-12-31",
            bands=["B01", "B02", "B3N"],  # VNIR bands only
            resolution=15,  # 15m resolution
            crs="EPSG:4326",
        )
        
        print(f"Status: {result['status']}")
        print(f"Bands: {result['bands']}")
        print(f"Resolution: {result['resolution']}m")
    except ImportError as e:
        print(f"Note: {e}")
        print("This is a demonstration of the API structure.")
    print()


def example_custom_config():
    """Example: Using MedianComputer class for more control."""
    print("Example: Custom Configuration")
    print("-" * 50)
    
    try:
        # Initialize computer with API key
        computer = MedianComputer(api_key="your-api-key-here")
        
        # Compute median with custom parameters
        result = computer.compute_median(
            sensor="sentinel2",
            bbox=[115.0, -32.0, 116.0, -31.0],
            start_date="2023-06-01",
            end_date="2023-08-31",
            bands=["B8", "B4", "B3"],  # NIR, Red, Green (false color)
            resolution=20,
            crs="EPSG:32750",  # UTM Zone 50S
        )
        
        print(f"Status: {result['status']}")
        print(f"Bands: {result['bands']}")
        print(f"DateTime: {result['datetime']}")
    except ImportError as e:
        print(f"Note: {e}")
        print("This is a demonstration of the API structure.")
    print()


if __name__ == "__main__":
    print("EarthOne Medians - Usage Examples")
    print("=" * 50)
    print()
    
    example_sentinel2()
    example_landsat()
    example_aster()
    example_custom_config()
    
    print("\nNote: These examples show the API structure.")
    print("Actual computation requires a valid EarthDaily API key.")
