# earthone-medians

Build temporal median composites for Sentinel-2, Landsat, and ASTER satellite imagery using the **EarthOne Platform** by EarthDaily with its serverless compute capability.

This package uses the official [earthdaily-earthone](https://github.com/earthdaily/earthone-python) Python client to access the EarthOne Platform.

## Three Approaches Available

This package provides **THREE** distinct approaches for computing composites:

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

### 3. 🌍 Bare Earth Modelling
**Best for:** Soil/mineral mapping, geology, agriculture, environmental monitoring  
**Based on:** Roberts et al. (2019) "Exposed soil and mineral map of the Australian continent"

```python
from earthone_medians import compute_bare_earth_landsat

result = compute_bare_earth_landsat(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="1990-01-01",
    end_date="2023-12-31",  # Multi-decadal time series recommended
    ndvi_threshold=0.3,  # NDVI threshold for bare earth classification
    compute_indices=True,  # Compute spectral indices for soil/mineral mapping
)
# Returns bare earth composite revealing soil and rock with minimal vegetation
```

📖 **See [TWO_APPROACHES.md](TWO_APPROACHES.md) for detailed comparison and usage guide**

⚙️ **See [SERVERLESS_LIMITS.md](SERVERLESS_LIMITS.md) for serverless compute quotas, limits, and best practices**

## Features

- **Multi-sensor support**: Sentinel-2, Landsat, and ASTER
- **Band selection**: Choose specific science bands, excluding non-useful bands like coastal aerosol
- **Bare Earth Modelling**: Implements the barest earth algorithm from Roberts et al. (2019)
- **Spectral Indices**: Computes soil and mineral indices (NDVI, BSI, Iron Oxide, Ferrous, Clay, Carbonate)
- **Configurable parameters**: 
  - Custom resolution
  - Custom CRS (coordinate reference system)
  - Start and end dates for time series
  - NDVI threshold for bare earth classification
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

# Bare Earth computation (Landsat recommended for multi-decadal analysis)
earthone-medians landsat \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "1990-01-01" \
    --end-date "2023-12-31" \
    --bare-earth \
    --ndvi-threshold 0.3

# Bare Earth with serverless compute for large areas
earthone-medians landsat \
    --bbox "115.0,-35.0,120.0,-30.0" \
    --start-date "1990-01-01" \
    --end-date "2023-12-31" \
    --bare-earth \
    --method serverless \
    --cpus 2.0 \
    --memory 4096

# List available bands
earthone-medians sentinel2 --list-bands
earthone-medians landsat --list-bands
earthone-medians aster --list-bands

# List spectral indices for bare earth soil/mineral mapping
earthone-medians sentinel2 --list-indices
```

## Bare Earth Modelling

This package implements **bare earth (barest earth) modelling** based on:

1. **Roberts, D., Wilford, J., & Ghattas, O. (2019).** "Exposed soil and mineral map of the Australian continent revealing the land at its barest." *Nature Communications*, 10, 5297.  
   DOI: [10.1038/s41467-019-13276-1](https://doi.org/10.1038/s41467-019-13276-1)

2. **Wilford, J., Roberts, D., & Thomas, M. (2021).** "Enhanced barest earth Landsat imagery for soil and lithological modelling." Extended Abstract.

### What is Bare Earth Modelling?

Bare earth modelling uses a **weighted geometric median** algorithm to identify the "barest" state of the landscape across a multi-decadal time series of satellite imagery. The approach:

- Minimizes the influence of clouds, shadows, and especially **vegetation**
- Reveals true **soil and rock reflectance** by favoring pixels with low vegetation cover
- Uses NDVI (Normalized Difference Vegetation Index) weighting to prioritize bare observations
- Produces composites ideal for **geology, soil mapping, mineral exploration, and agriculture**

### Python API - Bare Earth

```python
from earthone_medians import (
    compute_bare_earth_sentinel2,
    compute_bare_earth_landsat,
    compute_bare_earth_aster,
    get_spectral_indices_info,
    BARE_EARTH_REQUIRED_BANDS,
)

# Landsat bare earth (recommended for multi-decadal analysis)
result = compute_bare_earth_landsat(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="1990-01-01",
    end_date="2023-12-31",
    ndvi_threshold=0.3,  # Pixels with NDVI < 0.3 get higher weight
    compute_indices=True,  # Compute spectral indices
)

# Sentinel-2 bare earth (higher resolution, shorter time series)
result = compute_bare_earth_sentinel2(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2018-01-01",
    end_date="2023-12-31",
    resolution=10,
)

# View available spectral indices
indices = get_spectral_indices_info()
for name, info in indices.items():
    print(f"{name}: {info['description']}")

# View required bands for bare earth computation
print(BARE_EARTH_REQUIRED_BANDS["landsat"])  # ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
```

### Spectral Indices for Soil/Mineral Mapping

The bare earth module computes the following spectral indices:

| Index | Name | Description |
|-------|------|-------------|
| `ndvi` | Normalized Difference Vegetation Index | Used for vegetation masking and weighting |
| `bsi` | Bare Soil Index | Highlights bare soil areas |
| `iron_oxide` | Iron Oxide Ratio | Highlights iron oxide minerals (ferric iron) |
| `ferrous` | Ferrous Minerals Ratio | Highlights ferrous minerals |
| `clay_minerals` | Clay Minerals Ratio | Highlights clay/hydroxyl-bearing minerals |
| `carbonate` | Carbonate Index | Highlights carbonate minerals |

### Required Bands for Bare Earth

Each sensor requires specific bands for bare earth computation:

| Sensor | Required Bands |
|--------|---------------|
| Sentinel-2 | B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2) |
| Landsat | B2 (Blue), B3 (Green), B4 (Red), B5 (NIR), B6 (SWIR1), B7 (SWIR2) |
| ASTER | B01 (Green), B02 (Red), B3N (NIR), B04 (SWIR1), B05 (SWIR2)* |

*Note: ASTER lacks a blue band, so some spectral indices (e.g., iron oxide ratio) may not be computable.

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
- Bare earth band mappings and spectral indices

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

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
