"""Quick test of EarthOne API patterns."""
from earthdaily.earthone.catalog import Product, properties as p
from shapely.geometry import box

product = Product.get('esa:sentinel-2:l2a:v1')
print(f"Product: {product}")

bbox_geom = box(-70.05, -33.45, -69.95, -33.35)
print(f"BBox: {bbox_geom.bounds}")

search = product.images().filter(p.acquired >= '2023-01-01').filter(p.acquired < '2023-01-31').intersects(bbox_geom)
print("Search created, collecting...")

images = search.collect()
print(f"Found {len(images)} images")
