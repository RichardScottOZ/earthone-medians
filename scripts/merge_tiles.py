#!/usr/bin/env python
"""
Merge bare earth tiles into a single mosaic GeoTIFF.
"""
import argparse
from pathlib import Path
import rasterio
from rasterio.merge import merge
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Merge bare earth tiles into mosaic")
    parser.add_argument("input_dir", help="Directory containing tile GeoTIFFs")
    parser.add_argument("output", help="Output mosaic GeoTIFF path")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for tiles (default: *.tif)")
    parser.add_argument("--nodata", type=float, default=0, help="NoData value (default: 0)")
    parser.add_argument("--compress", default="lzw", help="Compression (default: lzw)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    tiles = sorted(input_dir.glob(args.pattern))
    
    if not tiles:
        print(f"No tiles found matching {args.pattern} in {input_dir}")
        return
    
    print(f"Found {len(tiles)} tiles")
    
    # Open all tiles
    src_files = [rasterio.open(t) for t in tiles]
    
    # Get band descriptions from first tile
    band_descriptions = [src_files[0].descriptions[i] for i in range(src_files[0].count)]
    
    print(f"Merging {len(src_files)} tiles...")
    mosaic, transform = merge(src_files, nodata=args.nodata)
    
    # Get metadata from first file
    profile = src_files[0].profile.copy()
    profile.update(
        driver='GTiff',
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        compress=args.compress,
        nodata=args.nodata,
        bigtiff='yes'  # For large mosaics
    )
    
    print(f"Writing mosaic: {mosaic.shape} to {args.output}")
    with rasterio.open(args.output, 'w', **profile) as dst:
        dst.write(mosaic)
        for i, desc in enumerate(band_descriptions):
            if desc:
                dst.set_band_description(i + 1, desc)
    
    # Close source files
    for src in src_files:
        src.close()
    
    print(f"Done: {args.output}")
    print(f"Size: {Path(args.output).stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
