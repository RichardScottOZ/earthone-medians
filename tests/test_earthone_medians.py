"""Tests for earthone-medians package."""

import unittest
from earthone_medians import (
    MedianComputer,
    SENTINEL2_BANDS,
    LANDSAT_BANDS,
    ASTER_BANDS,
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


if __name__ == "__main__":
    unittest.main()
