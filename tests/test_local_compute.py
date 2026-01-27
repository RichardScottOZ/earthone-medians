"""Test the compute logic locally to find errors."""
import numpy as np
from earthdaily.earthone.catalog import Product, properties as p
from earthdaily.earthone.geo import AOI
from shapely.geometry import box

collection = "esa:sentinel-2:l2a:v1"
bbox = [-70.05, -33.45, -69.95, -33.35]
start_date = "2023-01-01"
end_date = "2023-01-31"
bands = ["B2", "B3", "B4", "B8"]
resolution = 10
crs = "EPSG:4326"
max_cloud_cover = 20

# Convert resolution from meters to degrees (approximate at this latitude)
# ~111,000 meters per degree at equator, less at higher latitudes
resolution_deg = resolution / 111000.0  # Rough approximation

print("1. Creating bbox geometry...")
bbox_geom = box(*bbox)
minx, miny, maxx, maxy = bbox

print("2. Creating AOI...")
aoi = AOI(geometry=bbox_geom, crs=crs, resolution=resolution_deg)
print(f"   AOI: {aoi}")

print("3. Getting product...")
product = Product.get(collection)
print(f"   Product: {product}")

print("4. Searching images...")
search = product.images().filter(
    p.acquired >= start_date
).filter(
    p.acquired < end_date
).filter(
    p.cloud_fraction <= max_cloud_cover / 100.0
).intersects(bbox_geom)

images = search.collect()
num_images = len(images)
print(f"   Found {num_images} images")

if num_images == 0:
    print("No images found!")
    exit(1)

print("5. Stacking bands...")
band_str = " ".join(bands)
print(f"   Bands: {band_str}")
stack = images.stack(band_str, aoi)
print(f"   Stack shape: {stack.shape}")

print("6. Computing median...")
median_result = np.median(stack, axis=0)
print(f"   Median shape: {median_result.shape}")
print(f"   Median dtype: {median_result.dtype}")
print(f"   Median range: {median_result.min()} - {median_result.max()}")

print("\nSUCCESS - all steps completed!")
