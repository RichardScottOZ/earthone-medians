# EarthOne Medians - Feature Demonstration

This document demonstrates all the key features of the earthone-medians package using the **EarthOne Platform** by EarthDaily.

## Prerequisites

1. Register for EarthOne Platform access at https://earthdaily.com/earthone
2. Authenticate using `earthone auth login` or set environment variables

## Installation

```bash
pip install -e .
```

## Authentication

```bash
# Interactive login (recommended)
earthone auth login

# Or set environment variables
export EARTHONE_CLIENT_ID="your-client-id"
export EARTHONE_CLIENT_SECRET="your-client-secret"
```

## Feature 1: Multi-Sensor Support

The package supports three satellite sensors:
- **Sentinel-2** (10m default resolution)
- **Landsat** (30m default resolution)  
- **ASTER** (15m default resolution)

## Feature 2: Band Selection

### Viewing Available Bands

```bash
# List Sentinel-2 bands (excludes coastal aerosol B1)
earthone-medians sentinel2 --list-bands

# List Landsat bands (excludes coastal aerosol B1)
earthone-medians landsat --list-bands

# List ASTER bands
earthone-medians aster --list-bands
```

### Science Bands Only

The package automatically excludes non-science bands like coastal aerosol:
- Sentinel-2: B1 (coastal aerosol) is excluded, includes B2-B12 (except B9, B10)
- Landsat: B1 (coastal aerosol) is excluded, includes B2-B7
- ASTER: All VNIR and SWIR bands included

## Feature 3: Configurable Resolution

```python
from earthone_medians import compute_sentinel2_median

# Use custom resolution
result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    resolution=20,  # 20m instead of default 10m
)
```

Or via CLI:
```bash
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --resolution 20
```

## Feature 4: Configurable CRS

```python
from earthone_medians import compute_sentinel2_median

# Use UTM projection instead of default WGS84
result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    crs="EPSG:32750",  # UTM Zone 50S
)
```

Or via CLI:
```bash
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --crs "EPSG:32750"
```

## Feature 5: Date Range Selection

```python
from earthone_medians import compute_sentinel2_median

# Compute median for specific time period
result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-06-01",  # Winter season
    end_date="2023-08-31",
    bands=["B2", "B3", "B4", "B8"],
)
```

## Feature 6: Custom Band Selection

```python
from earthone_medians import compute_sentinel2_median

# Select specific bands
result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B8", "B4", "B3"],  # NIR, Red, Green for false color
)
```

Or via CLI:
```bash
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --bands "B8,B4,B3"
```

## Feature 7: EarthOne Platform Integration

The package uses the EarthOne Platform's serverless Compute API and Mosaic functions:

```python
from earthone_medians import MedianComputer

# Initialize (uses authenticated EarthOne session)
computer = MedianComputer()

# Compute median using serverless compute
result = computer.compute_median(
    sensor="sentinel2",
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
)
```

The computation happens on EarthOne's infrastructure:
- Searches the EarthOne Catalog for matching imagery
- Creates a Mosaic with median function
- Returns the computed result

Authentication is handled via:
```bash
# Method 1: Interactive login
earthone auth login

# Method 2: Environment variables
export EARTHONE_CLIENT_ID="your-client-id"
export EARTHONE_CLIENT_SECRET="your-client-secret"
```

## Complete Examples

### Example 1: Sentinel-2 True Color Composite

```python
from earthone_medians import compute_sentinel2_median

result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],  # Perth, Australia
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4"],  # Blue, Green, Red
    resolution=10,
    crs="EPSG:4326",
)
```

### Example 2: Landsat NDVI Analysis

```python
from earthone_medians import compute_landsat_median

result = compute_landsat_median(
    bbox=[-122.5, 37.5, -122.0, 38.0],  # San Francisco
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["B4", "B5"],  # Red, NIR for NDVI
    resolution=30,
    crs="EPSG:32610",  # UTM Zone 10N
)
```

### Example 3: ASTER Thermal Analysis

```python
from earthone_medians import compute_aster_median

result = compute_aster_median(
    bbox=[138.0, -35.0, 139.0, -34.0],  # Adelaide, Australia
    start_date="2020-01-01",
    end_date="2023-12-31",
    bands=["B01", "B02", "B3N", "B04", "B05"],  # VNIR + SWIR
    resolution=15,
)
```

## Testing

Run the test suite:

```bash
python -m unittest tests.test_earthone_medians -v
```

All tests validate:
- Configuration of sensor bands (excluding coastal aerosol)
- Resolution settings for each sensor
- CRS configuration
- Band selection validation
- API parameter validation
