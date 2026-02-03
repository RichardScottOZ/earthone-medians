#!/usr/bin/env python
"""
Bare Earth composite for South America Andes region.
Tiles the AOI and submits parallel serverless jobs.
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from earthdaily.earthone.catalog import Product, Blob, properties as p
from earthdaily.earthone.compute import Function
from earthdaily.earthone.geo import AOI
from shapely.geometry import box

# Andes AOI bounding box (from 4d-andes project)
ANDES_BBOX = [-81.8, -39.5, -56.3, 0.8]  # [minx, miny, maxx, maxy]

# Test subset - 2x2 degree chunk in central Chile/Argentina
TEST_BBOX = [-70.5, -34.0, -68.5, -32.0]

# GA Bare Earth bands (10 bands per Wilford et al. 2021)
BARE_EARTH_BANDS = [
    "blue", "green", "red",
    "red-edge", "red-edge-2", "red-edge-3",
    "nir", "red-edge-4",  # red-edge-4 = B8A (nir-narrow)
    "swir1", "swir2"
]

def generate_tiles(bbox, tile_size=0.5):
    """Generate tile bboxes covering the AOI."""
    minx, miny, maxx, maxy = bbox
    tiles = []
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            tile = [x, y, min(x + tile_size, maxx), min(y + tile_size, maxy)]
            tiles.append(tile)
            x += tile_size
        y += tile_size
    return tiles

def submit_tile_job(tile_id, tile, start_date, end_date, bands, resolution, memory, cpus, cloud_max, max_concurrent, retries, median_type, overlap):
    """Submit a single tile job and return job object."""
    minx, miny, maxx, maxy = tile
    bbox_geom = box(minx, miny, maxx, maxy).__geo_interface__
    resolution_deg = resolution / 111000.0
    
    def compute_bare_earth_tile(
        bbox, start_date, end_date, bands, resolution, cloud_max, median_type, overlap, crs="EPSG:4326"
    ):
        import numpy as np
        import io
        import traceback
        import rasterio
        from rasterio.transform import from_bounds
        from earthdaily.earthone.catalog import Product, Blob, properties as p
        from earthdaily.earthone.geo import AOI
        from shapely.geometry import box
        
        try:
            minx, miny, maxx, maxy = bbox
            # Expand bbox by overlap for processing, clip back later
            buf_minx, buf_miny = minx - overlap, miny - overlap
            buf_maxx, buf_maxy = maxx + overlap, maxy + overlap
            buffered_geom = box(buf_minx, buf_miny, buf_maxx, buf_maxy).__geo_interface__
            resolution_deg = resolution / 111000.0
            aoi = AOI(geometry=buffered_geom, crs=crs, resolution=resolution_deg)
            
            product = Product.get("esa:sentinel-2:l2a:v1")
            search = (product.images()
                .filter(p.acquired >= start_date)
                .filter(p.acquired < end_date)
                .filter(p.cloud_fraction <= cloud_max)
                .intersects(buffered_geom))
            images = search.collect()
            
            if len(images) == 0:
                return {"status": "no_images", "bbox": bbox}
            
            # Stack with NIR for NDVI weighting
            stack_bands = list(bands) if "nir" in bands else list(bands) + ["nir"]
            stack = images.stack(" ".join(stack_bands), aoi)
            data = np.ma.masked_invalid(stack.data)
            
            # Compute NDVI weights (favor low vegetation)
            nir_idx = stack_bands.index("nir")
            red_idx = stack_bands.index("red")
            nir = data[:, nir_idx].astype(np.float32)
            red = data[:, red_idx].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-10)
            weights = np.clip(1.0 - ndvi, 0.1, 1.0)
            
            # Compute NDSI for snow exclusion (favor non-snow)
            green_idx = stack_bands.index("green")
            swir1_idx = stack_bands.index("swir1")
            green = data[:, green_idx].astype(np.float32)
            swir1 = data[:, swir1_idx].astype(np.float32)
            ndsi = (green - swir1) / (green + swir1 + 1e-10)
            snow_weight = np.where(ndsi > 0.4, 0.1, 1.0)  # Penalize snow
            weights = weights * snow_weight
            
            # Keep only requested bands for median
            band_indices = [stack_bands.index(b) for b in bands]
            data_bands = data[:, band_indices, :, :].filled(np.nan)
            n_img, n_bands, h, w = [int(x) for x in data_bands.shape]
            
            if median_type == "geometric":
                # Mask out high-veg and snow observations (weight < 0.3)
                bad_obs = weights < 0.3  # (images, h, w)
                bad_obs_expanded = bad_obs[:, np.newaxis, :, :]  # (images, 1, h, w)
                data_bands = np.where(bad_obs_expanded, np.nan, data_bands)
                
                # Reshape: (images, bands, h, w) -> (h*w, images, bands)
                data_reshaped = data_bands.transpose(2, 3, 0, 1).reshape(h * w, n_img, n_bands)
                
                # Vectorized Weiszfeld geometric median across all pixels
                # data_reshaped: (pixels, images, bands)
                median = np.array(np.nanmedian(data_reshaped, axis=1), dtype=np.float32)  # initial guess (pixels, bands)
                
                for _ in range(50):  # max iterations
                    # Compute distances from each observation to current median
                    diff = data_reshaped - median[:, np.newaxis, :]  # (pixels, images, bands)
                    dists = np.sqrt(np.nansum(diff ** 2, axis=2))  # (pixels, images)
                    dists = np.maximum(dists, 1e-10)
                    
                    # Weights = 1/distance
                    wts = 1.0 / dists  # (pixels, images)
                    wts = np.where(np.isnan(data_reshaped[:, :, 0]), 0, wts)  # zero weight for NaN
                    
                    # Weighted average
                    wts_sum = wts.sum(axis=1, keepdims=True)  # (pixels, 1)
                    wts_norm = wts / np.maximum(wts_sum, 1e-10)  # (pixels, images)
                    new_median = np.array(np.nansum(data_reshaped * wts_norm[:, :, np.newaxis], axis=1), dtype=np.float32)
                    
                    # Check convergence
                    max_diff = float(np.nanmax(np.abs(new_median - median)))
                    if max_diff < 1e-5:
                        break
                    median = new_median
                
                # Reshape to output format
                median_result = median.reshape((h, w, n_bands)).transpose(2, 0, 1)
                median_result = np.nan_to_num(median_result, nan=0.0)
            else:
                # Basic: weighted mean approximation
                data_bands = np.ma.array(data_bands, mask=np.isnan(data_bands))
                weights_expanded = weights[:, np.newaxis, :, :]
                weighted_sum = np.ma.sum(data_bands * weights_expanded, axis=0)
                weight_sum = np.ma.sum(weights_expanded * ~data_bands.mask, axis=0)
                median_result = (weighted_sum / (weight_sum + 1e-10)).filled(0)
            
            # Clip back to original bbox if overlap was used
            if overlap > 0:
                full_h, full_w = median_result.shape[-2:]
                # Calculate pixel offsets for clipping
                px_overlap_x = int(round(overlap / resolution_deg))
                px_overlap_y = int(round(overlap / resolution_deg))
                # Safety check - don't clip more than available
                px_overlap_x = min(px_overlap_x, full_w // 4)
                px_overlap_y = min(px_overlap_y, full_h // 4)
                if px_overlap_x > 0 and px_overlap_y > 0:
                    y_start, y_end = px_overlap_y, full_h - px_overlap_y
                    x_start, x_end = px_overlap_x, full_w - px_overlap_x
                    median_result = median_result[:, y_start:y_end, x_start:x_end]
            
            # Save GeoTIFF
            num_bands = median_result.shape[0]
            height = int(median_result.shape[-2])
            width = int(median_result.shape[-1])
            transform = from_bounds(minx, miny, maxx, maxy, width, height)
            
            buffer = io.BytesIO()
            with rasterio.open(buffer, 'w', driver='GTiff',
                height=height, width=width, count=num_bands,
                dtype='float32', crs=crs, transform=transform, compress='lzw'
            ) as dst:
                for i in range(num_bands):
                    dst.write(median_result[i].astype('float32'), i + 1)
                    dst.set_band_description(i + 1, bands[i])
            
            buffer.seek(0)
            blob_name = f"bare_earth{'_geomedian' if median_type == 'geometric' else ''}_{resolution}m_{minx:.2f}_{miny:.2f}_{maxx:.2f}_{maxy:.2f}_{start_date}_{end_date}.tif"
            blob = Blob(name=blob_name)
            blob.upload_data(buffer.getvalue())
            
            return {
                "status": "success",
                "blob_id": blob.id,
                "blob_name": blob_name,
                "num_images": len(images),
                "shape": list(median_result.shape),
                "bbox": bbox
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "bbox": bbox}
    
    func = Function(
        compute_bare_earth_tile,
        name=f"bare-earth-v8-{tile_id}",
        image="python3.10:latest",
        cpus=cpus,
        memory=memory,
        timeout=3600,
        maximum_concurrency=max_concurrent,
        retry_count=retries,
        requirements=["earthdaily-earthone>=5.0.0", "numpy>=2.0.0", "shapely>=2.0.0", "rasterio>=1.3.0"]
    )
    
    job = func(
        bbox=tile,
        start_date=start_date,
        end_date=end_date,
        bands=bands,
        resolution=resolution,
        cloud_max=cloud_max,
        median_type=median_type,
        overlap=overlap
    )
    return job, tile_id, tile

def poll_job(job, tile_id, timeout=3600):
    """Poll job until complete."""
    start = time.time()
    while time.time() - start < timeout:
        job.refresh()
        elapsed = int(time.time() - start)
        print(f"  [{elapsed}s] {tile_id}: {job.status}")
        if job.status == "success":
            result = job.result()
            stats = getattr(job, 'statistics', None)
            if stats:
                stats = {"cpu": str(stats.cpu) if hasattr(stats, 'cpu') else None,
                         "memory": str(stats.memory) if hasattr(stats, 'memory') else None,
                         "network": str(stats.network) if hasattr(stats, 'network') else None}
            return {**result, "statistics": stats} if isinstance(result, dict) else result
        elif job.status == "failure":
            try:
                result = job.result()
                return {"status": "failure", "error": str(result)}
            except:
                return {"status": "failure", "error": "Job failed (no details)"}
        time.sleep(10)
    return {"status": "timeout"}

def main():
    parser = argparse.ArgumentParser(description="Bare Earth composite for Andes")
    parser.add_argument("--start", default="2019-01-01", help="Start date")
    parser.add_argument("--end", default="2024-01-01", help="End date")
    parser.add_argument("--tile-size", type=float, default=0.1, help="Tile size in degrees")
    parser.add_argument("--overlap", type=float, default=0.0, help="Tile overlap/buffer in degrees (e.g., 0.01)")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent jobs")
    parser.add_argument("--memory", type=int, default=16384, help="Memory per job (MB)")
    parser.add_argument("--cpus", type=float, default=4.0, help="CPUs per job")
    parser.add_argument("--resolution", type=int, default=10, help="Resolution in meters")
    parser.add_argument("--cloud", type=float, default=0.1, help="Max cloud fraction 0-1")
    parser.add_argument("--median-type", choices=["basic", "geometric"], default="basic", help="Median type: basic (per-band) or geometric (spectral)")
    parser.add_argument("--retries", type=int, default=0, help="Retry count on failure")
    parser.add_argument("--output-dir", default="./bare_earth_tiles", help="Local output directory for tiles")
    parser.add_argument("--s3-bucket", help="S3 bucket for output (optional, e.g. s3://my-bucket/bare-earth/)")
    parser.add_argument("--download", action="store_true", help="Download tiles as they complete")
    parser.add_argument("--resume", action="store_true", help="Resume from progress file")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed tiles when resuming")
    parser.add_argument("--dry-run", action="store_true", help="Show tiles without running")
    parser.add_argument("--test", action="store_true", help="Use small test AOI instead of full Andes")
    parser.add_argument("--bbox", type=str, help="Custom bbox: minx,miny,maxx,maxy (e.g. -71.75,-37.75,-69,-32.75)")
    parser.add_argument("--limit", type=int, help="Limit number of tiles to process")
    args = parser.parse_args()

    if args.bbox:
        bbox = [float(x) for x in args.bbox.split(",")]
    elif args.test:
        bbox = TEST_BBOX
    else:
        bbox = ANDES_BBOX

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progress.json"

    tiles = generate_tiles(bbox, args.tile_size)
    print(f"AOI: {bbox}")
    print(f"Total tiles: {len(tiles)} ({args.tile_size}° grid)")

    if args.dry_run:
        print(f"Would process {len(tiles)} tiles from {args.start} to {args.end}")
        print(f"Estimated time at {args.max_concurrent} concurrent: {len(tiles) * 4 / args.max_concurrent / 60:.1f} hours")
        return

    # Load progress if resuming
    completed = {}
    if args.resume and progress_file.exists():
        with open(progress_file) as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} tiles in progress file")

    # Filter out completed tiles (keep failed ones if --retry-failed)
    def should_skip(tile_id):
        if tile_id not in completed:
            return False
        if args.retry_failed and completed[tile_id].get("status") != "success":
            return False
        return True
    
    pending_tiles = [(f"tile_{i:04d}", tile) for i, tile in enumerate(tiles) 
                     if not should_skip(f"tile_{i:04d}")]
    if args.limit:
        pending_tiles = pending_tiles[:args.limit]
    print(f"Tiles to process: {len(pending_tiles)}")

    # Submit and poll jobs in parallel
    active_jobs = {}  # tile_id -> (job, tile)
    
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        tile_iter = iter(pending_tiles)
        futures = {}
        
        # Initial batch
        for _ in range(min(args.max_concurrent, len(pending_tiles))):
            try:
                tile_id, tile = next(tile_iter)
                job, tid, t = submit_tile_job(tile_id, tile, args.start, args.end, 
                    BARE_EARTH_BANDS, args.resolution, args.memory, args.cpus, args.cloud, args.max_concurrent, args.retries, args.median_type, args.overlap)
                future = executor.submit(poll_job, job, tid)
                futures[future] = (tid, t, job.id)
                print(f"Submitted {tid}: {t}")
            except StopIteration:
                break
        
        # Process as jobs complete
        while futures:
            done_futures = [f for f in futures if f.done()]
            
            for future in done_futures:
                tile_id, tile, job_id = futures.pop(future)
                result = future.result()
                
                completed[tile_id] = {
                    "bbox": tile,
                    "job_id": job_id,
                    **result
                }
                
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"✓ {tile_id}: {result.get('num_images')} images")
                    
                    # Download tile if requested
                    if args.download and result.get("blob_id"):
                        try:
                            from earthdaily.earthone.catalog import Blob
                            blob = Blob.get(id=result["blob_id"])
                            data = blob.get_data(id=result["blob_id"])
                            
                            if args.s3_bucket:
                                import boto3
                                s3 = boto3.client('s3')
                                bucket = args.s3_bucket.replace("s3://", "").split("/")[0]
                                prefix = "/".join(args.s3_bucket.replace("s3://", "").split("/")[1:])
                                key = f"{prefix}{tile_id}.tif" if prefix else f"{tile_id}.tif"
                                s3.put_object(Bucket=bucket, Key=key, Body=data)
                                completed[tile_id]["s3_path"] = f"s3://{bucket}/{key}"
                                print(f"  → Uploaded to s3://{bucket}/{key}")
                            else:
                                local_path = output_dir / f"{tile_id}.tif"
                                with open(local_path, 'wb') as f:
                                    f.write(data)
                                completed[tile_id]["local_path"] = str(local_path)
                                print(f"  → Saved to {local_path}")
                        except Exception as e:
                            print(f"  ⚠ Download failed: {e}")
                else:
                    print(f"✗ {tile_id}: {status} - {result.get('error', '')[:50]}")
                
                # Save progress (with retry for file lock)
                for attempt in range(3):
                    try:
                        with open(progress_file, "w") as f:
                            json.dump(completed, f, indent=2)
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(1)
                        else:
                            print(f"  ⚠ Could not save progress (file locked)")
                
                # Submit next tile
                try:
                    next_tile_id, next_tile = next(tile_iter)
                    job, tid, t = submit_tile_job(next_tile_id, next_tile, args.start, args.end,
                        BARE_EARTH_BANDS, args.resolution, args.memory, args.cpus, args.cloud, args.max_concurrent, args.retries, args.median_type, args.overlap)
                    new_future = executor.submit(poll_job, job, tid)
                    futures[new_future] = (tid, t, job.id)
                    print(f"Submitted {tid}: {t}")
                except StopIteration:
                    pass
            
            if futures:
                time.sleep(5)

    success = sum(1 for v in completed.values() if v.get("status") == "success")
    print(f"\nCompleted: {success}/{len(tiles)} tiles")
    print(f"Progress: {progress_file}")

if __name__ == "__main__":
    main()
