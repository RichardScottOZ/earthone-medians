"""Tests for earthone-medians package."""

import unittest
from earthone_medians import (
    MedianComputer,
    SENTINEL2_BANDS,
    LANDSAT_BANDS,
    ASTER_BANDS,
    BareEarthComputer,
    ServerlessBareEarthComputer,
    SPECTRAL_INDICES,
    BARE_EARTH_BAND_MAPPINGS,
    BARE_EARTH_REQUIRED_BANDS,
    get_spectral_indices_info,
    get_bare_earth_band_mappings,
)
from earthone_medians.config import SENSOR_CONFIGS


class TestConfiguration(unittest.TestCase):
    """Test configuration constants."""

    def test_sentinel2_bands(self):
        """Test Sentinel-2 bands configuration."""
        self.assertIn("B2", SENTINEL2_BANDS)
        self.assertIn("B3", SENTINEL2_BANDS)
        self.assertIn("B4", SENTINEL2_BANDS)
        self.assertIn("B8", SENTINEL2_BANDS)
        # Verify coastal aerosol (B1) is excluded
        self.assertNotIn("B1", SENTINEL2_BANDS)

    def test_landsat_bands(self):
        """Test Landsat bands configuration."""
        self.assertIn("B2", LANDSAT_BANDS)
        self.assertIn("B3", LANDSAT_BANDS)
        self.assertIn("B4", LANDSAT_BANDS)
        self.assertIn("B5", LANDSAT_BANDS)
        # Verify coastal aerosol (B1) is excluded
        self.assertNotIn("B1", LANDSAT_BANDS)

    def test_aster_bands(self):
        """Test ASTER bands configuration."""
        self.assertIn("B01", ASTER_BANDS)
        self.assertIn("B02", ASTER_BANDS)
        self.assertIn("B3N", ASTER_BANDS)

    def test_sensor_configs(self):
        """Test sensor configurations."""
        self.assertIn("sentinel2", SENSOR_CONFIGS)
        self.assertIn("landsat", SENSOR_CONFIGS)
        self.assertIn("aster", SENSOR_CONFIGS)
        
        # Verify each sensor has required keys
        for sensor in ["sentinel2", "landsat", "aster"]:
            config = SENSOR_CONFIGS[sensor]
            self.assertIn("collection", config)
            self.assertIn("bands", config)
            self.assertIn("default_resolution", config)


class TestMedianComputer(unittest.TestCase):
    """Test MedianComputer class."""

    def test_initialization(self):
        """Test MedianComputer initialization."""
        computer = MedianComputer(api_key="test-key")
        self.assertEqual(computer.api_key, "test-key")

    def test_invalid_sensor(self):
        """Test invalid sensor raises error."""
        computer = MedianComputer()
        with self.assertRaises(ValueError):
            computer.compute_median(
                sensor="invalid",
                bbox=[0, 0, 1, 1],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

    def test_invalid_bands(self):
        """Test invalid bands raise error."""
        computer = MedianComputer()
        with self.assertRaises(ValueError):
            computer.compute_median(
                sensor="sentinel2",
                bbox=[0, 0, 1, 1],
                start_date="2023-01-01",
                end_date="2023-12-31",
                bands=["INVALID_BAND"],
            )

    def test_valid_parameters(self):
        """Test valid parameters are accepted."""
        computer = MedianComputer()
        # This will fail on API call but should pass validation
        try:
            computer.compute_median(
                sensor="sentinel2",
                bbox=[115.0, -32.0, 116.0, -31.0],
                start_date="2023-01-01",
                end_date="2023-12-31",
                bands=["B2", "B3", "B4"],
                resolution=10,
                crs="EPSG:4326",
            )
        except ImportError:
            # Expected when earthdaily is not installed
            pass


class TestBandSelection(unittest.TestCase):
    """Test band selection functionality."""

    def test_sentinel2_excludes_coastal_aerosol(self):
        """Test that Sentinel-2 excludes coastal aerosol (B1)."""
        self.assertNotIn("B1", SENTINEL2_BANDS)
        
    def test_landsat_excludes_coastal_aerosol(self):
        """Test that Landsat excludes coastal aerosol (B1)."""
        self.assertNotIn("B1", LANDSAT_BANDS)

    def test_default_bands(self):
        """Test that default bands are used when none specified."""
        computer = MedianComputer()
        # Verify sentinel2 has expected science bands
        s2_bands = SENSOR_CONFIGS["sentinel2"]["bands"]
        self.assertGreater(len(s2_bands), 5)
        
        # Verify landsat has expected science bands
        landsat_bands = SENSOR_CONFIGS["landsat"]["bands"]
        self.assertGreater(len(landsat_bands), 4)


class TestResolutionAndCRS(unittest.TestCase):
    """Test resolution and CRS configuration."""

    def test_default_resolutions(self):
        """Test default resolutions for each sensor."""
        self.assertEqual(SENSOR_CONFIGS["sentinel2"]["default_resolution"], 10)
        self.assertEqual(SENSOR_CONFIGS["landsat"]["default_resolution"], 30)
        self.assertEqual(SENSOR_CONFIGS["aster"]["default_resolution"], 15)

    def test_band_resolutions(self):
        """Test individual band resolutions."""
        # Sentinel-2 B2 should be 10m
        self.assertEqual(SENTINEL2_BANDS["B2"]["resolution"], 10)
        # Landsat B2 should be 30m
        self.assertEqual(LANDSAT_BANDS["B2"]["resolution"], 30)
        # ASTER B01 should be 15m
        self.assertEqual(ASTER_BANDS["B01"]["resolution"], 15)


class TestBareEarthConfiguration(unittest.TestCase):
    """Test bare earth configuration and band mappings."""

    def test_bare_earth_band_mappings_exist(self):
        """Test that bare earth band mappings exist for all sensors."""
        self.assertIn("sentinel2", BARE_EARTH_BAND_MAPPINGS)
        self.assertIn("landsat", BARE_EARTH_BAND_MAPPINGS)
        self.assertIn("aster", BARE_EARTH_BAND_MAPPINGS)

    def test_bare_earth_required_bands(self):
        """Test required bare earth bands for each sensor."""
        # Sentinel-2 requires 6 bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
        self.assertEqual(len(BARE_EARTH_REQUIRED_BANDS["sentinel2"]), 6)
        # Landsat requires 6 bands
        self.assertEqual(len(BARE_EARTH_REQUIRED_BANDS["landsat"]), 6)
        # ASTER requires 5 bands (no blue band)
        self.assertEqual(len(BARE_EARTH_REQUIRED_BANDS["aster"]), 5)

    def test_sentinel2_bare_earth_bands(self):
        """Test Sentinel-2 bare earth band mapping."""
        mapping = BARE_EARTH_BAND_MAPPINGS["sentinel2"]
        self.assertEqual(mapping["blue"], "B2")
        self.assertEqual(mapping["green"], "B3")
        self.assertEqual(mapping["red"], "B4")
        self.assertEqual(mapping["nir"], "B8")
        self.assertEqual(mapping["swir1"], "B11")
        self.assertEqual(mapping["swir2"], "B12")

    def test_landsat_bare_earth_bands(self):
        """Test Landsat bare earth band mapping."""
        mapping = BARE_EARTH_BAND_MAPPINGS["landsat"]
        self.assertEqual(mapping["blue"], "B2")
        self.assertEqual(mapping["green"], "B3")
        self.assertEqual(mapping["red"], "B4")
        self.assertEqual(mapping["nir"], "B5")
        self.assertEqual(mapping["swir1"], "B6")
        self.assertEqual(mapping["swir2"], "B7")

    def test_aster_bare_earth_bands(self):
        """Test ASTER bare earth band mapping (no blue band)."""
        mapping = BARE_EARTH_BAND_MAPPINGS["aster"]
        self.assertNotIn("blue", mapping)  # ASTER has no blue band
        self.assertEqual(mapping["green"], "B01")
        self.assertEqual(mapping["red"], "B02")
        self.assertEqual(mapping["nir"], "B3N")
        self.assertEqual(mapping["swir1"], "B04")
        self.assertEqual(mapping["swir2"], "B05")


class TestSpectralIndices(unittest.TestCase):
    """Test spectral indices configuration."""

    def test_spectral_indices_exist(self):
        """Test that expected spectral indices are defined."""
        expected_indices = ["ndvi", "bsi", "iron_oxide", "ferrous", "clay_minerals", "carbonate"]
        for index_name in expected_indices:
            self.assertIn(index_name, SPECTRAL_INDICES)

    def test_spectral_index_structure(self):
        """Test spectral index information structure."""
        for name, info in SPECTRAL_INDICES.items():
            self.assertIn("name", info)
            self.assertIn("formula", info)
            self.assertIn("description", info)
            self.assertIn("range", info)

    def test_get_spectral_indices_info(self):
        """Test get_spectral_indices_info function."""
        info = get_spectral_indices_info()
        self.assertEqual(info, SPECTRAL_INDICES)

    def test_ndvi_index(self):
        """Test NDVI spectral index configuration."""
        ndvi = SPECTRAL_INDICES["ndvi"]
        self.assertEqual(ndvi["name"], "Normalized Difference Vegetation Index")
        self.assertEqual(ndvi["range"], (-1.0, 1.0))

    def test_bsi_index(self):
        """Test Bare Soil Index configuration."""
        bsi = SPECTRAL_INDICES["bsi"]
        self.assertEqual(bsi["name"], "Bare Soil Index")


class TestBareEarthComputer(unittest.TestCase):
    """Test BareEarthComputer class."""

    def test_initialization(self):
        """Test BareEarthComputer initialization."""
        computer = BareEarthComputer()
        self.assertEqual(computer.ndvi_threshold, 0.3)
        self.assertEqual(computer.vegetation_weight_method, "inverse_ndvi")

    def test_initialization_custom_params(self):
        """Test BareEarthComputer with custom parameters."""
        computer = BareEarthComputer(
            ndvi_threshold=0.25,
            vegetation_weight_method="exponential"
        )
        self.assertEqual(computer.ndvi_threshold, 0.25)
        self.assertEqual(computer.vegetation_weight_method, "exponential")

    def test_get_required_bands(self):
        """Test get_required_bands method."""
        computer = BareEarthComputer()
        
        sentinel2_bands = computer.get_required_bands("sentinel2")
        self.assertEqual(sentinel2_bands, ["B2", "B3", "B4", "B8", "B11", "B12"])
        
        landsat_bands = computer.get_required_bands("landsat")
        self.assertEqual(landsat_bands, ["B2", "B3", "B4", "B5", "B6", "B7"])
        
        aster_bands = computer.get_required_bands("aster")
        self.assertEqual(aster_bands, ["B01", "B02", "B3N", "B04", "B05"])

    def test_get_required_bands_invalid_sensor(self):
        """Test get_required_bands with invalid sensor."""
        computer = BareEarthComputer()
        with self.assertRaises(ValueError):
            computer.get_required_bands("invalid_sensor")

    def test_invalid_sensor(self):
        """Test invalid sensor raises error."""
        computer = BareEarthComputer()
        with self.assertRaises(ValueError):
            computer.compute_bare_earth(
                sensor="invalid",
                bbox=[0, 0, 1, 1],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )

    def test_invalid_bands(self):
        """Test invalid bands raise error."""
        computer = BareEarthComputer()
        with self.assertRaises(ValueError):
            computer.compute_bare_earth(
                sensor="sentinel2",
                bbox=[0, 0, 1, 1],
                start_date="2023-01-01",
                end_date="2023-12-31",
                bands=["INVALID_BAND"],
            )

    def test_vegetation_weight_inverse_ndvi(self):
        """Test vegetation weight calculation with inverse NDVI method."""
        computer = BareEarthComputer(vegetation_weight_method="inverse_ndvi")
        
        # High vegetation (high NDVI) should have low weight
        weight_high_veg = computer.compute_vegetation_weight(0.8)
        # Low vegetation (low NDVI) should have high weight
        weight_low_veg = computer.compute_vegetation_weight(0.1)
        
        self.assertLess(weight_high_veg, weight_low_veg)
        self.assertGreaterEqual(weight_low_veg, 0.0)
        self.assertLessEqual(weight_low_veg, 1.0)

    def test_vegetation_weight_exponential(self):
        """Test vegetation weight calculation with exponential method."""
        computer = BareEarthComputer(vegetation_weight_method="exponential")
        
        weight_high_veg = computer.compute_vegetation_weight(0.8)
        weight_low_veg = computer.compute_vegetation_weight(0.1)
        
        self.assertLess(weight_high_veg, weight_low_veg)

    def test_compute_spectral_index_ndvi(self):
        """Test NDVI spectral index computation."""
        computer = BareEarthComputer()
        
        # Test NDVI calculation for Sentinel-2
        band_values = {"B4": 0.1, "B8": 0.4}  # Red=0.1, NIR=0.4
        ndvi = computer.compute_spectral_index("ndvi", band_values, "sentinel2")
        
        expected_ndvi = (0.4 - 0.1) / (0.4 + 0.1)  # = 0.6
        self.assertAlmostEqual(ndvi, expected_ndvi, places=5)

    def test_compute_spectral_index_iron_oxide(self):
        """Test iron oxide index computation."""
        computer = BareEarthComputer()
        
        # Test iron oxide ratio for Sentinel-2
        band_values = {"B2": 0.1, "B4": 0.3}  # Blue=0.1, Red=0.3
        iron_oxide = computer.compute_spectral_index("iron_oxide", band_values, "sentinel2")
        
        expected = 0.3 / 0.1  # = 3.0
        self.assertAlmostEqual(iron_oxide, expected, places=5)

    def test_compute_spectral_index_missing_bands(self):
        """Test spectral index returns None when required bands are missing."""
        computer = BareEarthComputer()
        
        # NDVI without NIR band should return None
        band_values = {"B4": 0.1}  # Only Red, no NIR
        ndvi = computer.compute_spectral_index("ndvi", band_values, "sentinel2")
        self.assertIsNone(ndvi)
        
        # Iron oxide without blue band should return None
        band_values = {"B4": 0.3}  # Only Red, no Blue
        iron_oxide = computer.compute_spectral_index("iron_oxide", band_values, "sentinel2")
        self.assertIsNone(iron_oxide)
        
        # ASTER iron oxide should return None (ASTER has no blue band)
        band_values = {"B02": 0.3}  # Red only for ASTER
        iron_oxide = computer.compute_spectral_index("iron_oxide", band_values, "aster")
        self.assertIsNone(iron_oxide)

    def test_valid_parameters(self):
        """Test valid parameters are accepted."""
        computer = BareEarthComputer()
        try:
            computer.compute_bare_earth(
                sensor="sentinel2",
                bbox=[115.0, -32.0, 116.0, -31.0],
                start_date="2023-01-01",
                end_date="2023-12-31",
                resolution=10,
                crs="EPSG:4326",
            )
        except ImportError:
            # Expected when earthdaily is not installed
            pass


class TestGetBareEarthBandMappings(unittest.TestCase):
    """Test get_bare_earth_band_mappings function."""

    def test_get_sentinel2_mapping(self):
        """Test getting Sentinel-2 band mapping."""
        mapping = get_bare_earth_band_mappings("sentinel2")
        self.assertEqual(mapping, BARE_EARTH_BAND_MAPPINGS["sentinel2"])

    def test_get_landsat_mapping(self):
        """Test getting Landsat band mapping."""
        mapping = get_bare_earth_band_mappings("landsat")
        self.assertEqual(mapping, BARE_EARTH_BAND_MAPPINGS["landsat"])

    def test_get_aster_mapping(self):
        """Test getting ASTER band mapping."""
        mapping = get_bare_earth_band_mappings("aster")
        self.assertEqual(mapping, BARE_EARTH_BAND_MAPPINGS["aster"])

    def test_invalid_sensor(self):
        """Test invalid sensor raises error."""
        with self.assertRaises(ValueError):
            get_bare_earth_band_mappings("invalid_sensor")


if __name__ == "__main__":
    unittest.main()
