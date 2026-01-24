"""Configuration for satellite sensors and their bands."""

# Sentinel-2 useful science bands (excluding coastal aerosol B1)
SENTINEL2_BANDS = {
    "B2": {"name": "Blue", "resolution": 10, "wavelength": "490 nm"},
    "B3": {"name": "Green", "resolution": 10, "wavelength": "560 nm"},
    "B4": {"name": "Red", "resolution": 10, "wavelength": "665 nm"},
    "B5": {"name": "Red Edge 1", "resolution": 20, "wavelength": "705 nm"},
    "B6": {"name": "Red Edge 2", "resolution": 20, "wavelength": "740 nm"},
    "B7": {"name": "Red Edge 3", "resolution": 20, "wavelength": "783 nm"},
    "B8": {"name": "NIR", "resolution": 10, "wavelength": "842 nm"},
    "B8A": {"name": "Red Edge 4", "resolution": 20, "wavelength": "865 nm"},
    "B11": {"name": "SWIR 1", "resolution": 20, "wavelength": "1610 nm"},
    "B12": {"name": "SWIR 2", "resolution": 20, "wavelength": "2190 nm"},
}

# Landsat useful science bands (excluding coastal aerosol)
LANDSAT_BANDS = {
    "B2": {"name": "Blue", "resolution": 30, "wavelength": "450-510 nm"},
    "B3": {"name": "Green", "resolution": 30, "wavelength": "530-590 nm"},
    "B4": {"name": "Red", "resolution": 30, "wavelength": "640-670 nm"},
    "B5": {"name": "NIR", "resolution": 30, "wavelength": "850-880 nm"},
    "B6": {"name": "SWIR 1", "resolution": 30, "wavelength": "1570-1650 nm"},
    "B7": {"name": "SWIR 2", "resolution": 30, "wavelength": "2110-2290 nm"},
}

# ASTER useful science bands
ASTER_BANDS = {
    "B01": {"name": "Green", "resolution": 15, "wavelength": "520-600 nm", "subsystem": "VNIR"},
    "B02": {"name": "Red", "resolution": 15, "wavelength": "630-690 nm", "subsystem": "VNIR"},
    "B3N": {"name": "NIR", "resolution": 15, "wavelength": "760-860 nm", "subsystem": "VNIR"},
    "B04": {"name": "SWIR 1", "resolution": 30, "wavelength": "1600-1700 nm", "subsystem": "SWIR"},
    "B05": {"name": "SWIR 2", "resolution": 30, "wavelength": "2145-2185 nm", "subsystem": "SWIR"},
    "B06": {"name": "SWIR 3", "resolution": 30, "wavelength": "2185-2225 nm", "subsystem": "SWIR"},
    "B07": {"name": "SWIR 4", "resolution": 30, "wavelength": "2235-2285 nm", "subsystem": "SWIR"},
    "B08": {"name": "SWIR 5", "resolution": 30, "wavelength": "2295-2365 nm", "subsystem": "SWIR"},
    "B09": {"name": "SWIR 6", "resolution": 30, "wavelength": "2360-2430 nm", "subsystem": "SWIR"},
}

# Sensor configurations
SENSOR_CONFIGS = {
    "sentinel2": {
        "collection": "sentinel-2-l2a",
        "bands": SENTINEL2_BANDS,
        "default_resolution": 10,
    },
    "landsat": {
        "collection": "landsat-c2l2",
        "bands": LANDSAT_BANDS,
        "default_resolution": 30,
    },
    "aster": {
        "collection": "aster-l1t",
        "bands": ASTER_BANDS,
        "default_resolution": 15,
    },
}

# Default CRS
DEFAULT_CRS = "EPSG:4326"
