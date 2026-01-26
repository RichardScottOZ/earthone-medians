# earthone-medians

Build temporal median composites for Sentinel-2, Landsat, and ASTER satellite imagery using the **EarthOne Platform** by EarthDaily with its serverless compute capability.

This package uses the official [earthdaily-earthone](https://github.com/earthdaily/earthone-python) Python client to access the EarthOne Platform.

## Two Approaches Available

This package provides **TWO** distinct approaches for computing medians:

### 1. 🚀 Workbench/Interactive (Dynamic Compute)
**Best for:** Interactive analysis, Jupyter notebooks, visualization  
**Uses:** `earthdaily.earthone.dynamic_compute.Mosaic`

```python
from earthone_medians import compute_sentinel2_median_workbench

result = compute_sentinel2_median_workbench(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],
)
# Returns Mosaic object for immediate visualization
```

### 2. ⚡ Serverless Compute (Batch Processing)
**Best for:** Production workflows, large-scale batch processing  
**Uses:** `earthdaily.earthone.compute.Function`

```python
from earthone_medians import compute_sentinel2_median_serverless

result = compute_sentinel2_median_serverless(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],
    cpus=2.0,  # Configure resources
    memory=4096,
)
# Submits batch job to EarthOne compute infrastructure
```

📖 **See [TWO_APPROACHES.md](TWO_APPROACHES.md) for detailed comparison and usage guide**

⚙️ **See [SERVERLESS_LIMITS.md](SERVERLESS_LIMITS.md) for serverless compute quotas, limits, and best practices**

## Features

- **Multi-sensor support**: Sentinel-2, Landsat, and ASTER
- **Band selection**: Choose specific science bands, excluding non-useful bands like coastal aerosol
- **Configurable parameters**: 
  - Custom resolution
  - Custom CRS (coordinate reference system)
  - Start and end dates for time series
- **Serverless compute**: Leverages EarthOne Platform's Compute API and Mosaic functions
- **Simple API**: Easy-to-use Python functions and CLI interface

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

**Note:** You must be a registered customer with access to the EarthOne Platform. Visit [https://earthdaily.com/earthone](https://earthdaily.com/earthone) to request access.

## Authentication

Before using this package, authenticate with the EarthOne Platform:

```bash
# Interactive login
earthone auth login
```

Or set environment variables:

```bash
export EARTHONE_CLIENT_ID="your-client-id"
export EARTHONE_CLIENT_SECRET="your-client-secret"
```

## Quick Start

### Python API

#### Workbench/Interactive (Default)
```python
from earthone_medians import compute_sentinel2_median

# Uses workbench approach by default (fast, interactive)
result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],  # [min_lon, min_lat, max_lon, max_lat]
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],  # Blue, Green, Red, NIR
    resolution=10,  # 10m resolution
    crs="EPSG:4326",
)
```

#### Serverless Compute (Production)
```python
from earthone_medians import compute_sentinel2_median_serverless

# Uses serverless batch processing (scalable, production)
result = compute_sentinel2_median_serverless(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],
    cpus=2.0,
    memory=4096,
)
```

### Command Line Interface

```bash
# Workbench approach (default, fast for exploration)
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --bands "B2,B3,B4,B8" \
    --resolution 10 \
    --crs "EPSG:4326" \
    --output result.json

# Serverless compute (for production batch processing)
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --method serverless \
    --cpus 2.0 \
    --memory 4096

# List available bands for a sensor
earthone-medians sentinel2 --list-bands
```

## Supported Sensors

### Sentinel-2

Science bands (coastal aerosol B1 excluded):
- B2 (Blue), B3 (Green), B4 (Red)
- B5, B6, B7 (Red Edge)
- B8, B8A (NIR)
- B11, B12 (SWIR)

Default resolution: 10m

### Landsat

Science bands (coastal aerosol excluded):
- B2 (Blue), B3 (Green), B4 (Red)
- B5 (NIR)
- B6, B7 (SWIR)

Default resolution: 30m

### ASTER

All VNIR and SWIR bands:
- B01 (Green), B02 (Red), B3N (NIR)
- B04-B09 (SWIR)

Default resolution: 15m

## API Usage Examples

### Sentinel-2 Median

```python
from earthone_medians import compute_sentinel2_median

result = compute_sentinel2_median(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],
    resolution=10,
    crs="EPSG:4326"
)
```

### Landsat Median

```python
from earthone_medians import compute_landsat_median

result = compute_landsat_median(
    bbox=[-122.5, 37.5, -122.0, 38.0],
    start_date="2022-01-01",
    end_date="2022-12-31",
    resolution=30,
    crs="EPSG:32610"
)
```

### ASTER Median

```python
from earthone_medians import compute_aster_median

result = compute_aster_median(
    bbox=[138.0, -35.0, 139.0, -34.0],
    start_date="2020-01-01",
    end_date="2023-12-31",
    bands=["B01", "B02", "B3N"],
    resolution=15,
    crs="EPSG:4326"
)
```

### Advanced Usage

```python
from earthone_medians import MedianComputer

# Initialize with API key
computer = MedianComputer(api_key="your-api-key-here")

# Compute median with custom parameters
result = computer.compute_median(
    sensor="sentinel2",
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-06-01",
    end_date="2023-08-31",
    bands=["B8", "B4", "B3"],
    resolution=20,
    crs="EPSG:32750"
)
```

## CLI Examples

```bash
# Sentinel-2 with specific bands
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --bands "B2,B3,B4" \
    --resolution 10

# Landsat with default bands
earthone-medians landsat \
    --bbox "-122.5,37.5,-122.0,38.0" \
    --start-date "2022-01-01" \
    --end-date "2022-12-31" \
    --crs "EPSG:32610"

# ASTER VNIR bands only
earthone-medians aster \
    --bbox "138.0,-35.0,139.0,-34.0" \
    --start-date "2020-01-01" \
    --end-date "2023-12-31" \
    --bands "B01,B02,B3N"

# List available bands
earthone-medians sentinel2 --list-bands
earthone-medians landsat --list-bands
earthone-medians aster --list-bands
```

## Authentication

Set your EarthOne credentials as environment variables:

```bash
export EARTHONE_CLIENT_ID="your-client-id"
export EARTHONE_CLIENT_SECRET="your-client-secret"
```

Or use interactive login:

```bash
earthone auth login
```

Then use the package without needing to pass credentials:

```python
result = compute_sentinel2_median(...)  # Uses authenticated session
```

```bash
earthone-medians sentinel2 ...  # Uses authenticated session
```

## Configuration

The package includes predefined configurations for each sensor in `config.py`:

- Band definitions with wavelengths and resolutions
- Default resolutions per sensor
- Collection names for EarthDaily API

You can customize these or use them as references for your analysis.

## Requirements

- Python >= 3.10
- earthdaily-earthone >= 5.0.0
- earthdaily-earthone-dynamic-compute >= 0.1.0
- numpy >= 2.0.0
- pandas >= 1.3.0
- geopandas >= 0.10.0
- Access to EarthOne Platform (register at https://earthdaily.com/earthone)

## License



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
