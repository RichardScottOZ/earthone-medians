# EarthOne Serverless Compute - Quotas, Limits & Best Practices

## Overview

EarthOne Platform enforces specific quotas and limits on its Compute API to ensure stability and fair resource allocation. Understanding these limits is crucial for planning large-scale batch processing workflows.

## Platform-Wide Limits

### Concurrent Jobs
- **Maximum Concurrent Invocations:** 1,000 concurrent function invocations per user
- This means you can have up to 1,000 compute jobs running in parallel across all your functions
- When this limit is reached, new job submissions will be queued or rejected

### Function-Specific Limits
When creating a Compute Function, you can (and should) specify:

```python
from earthdaily.earthone.compute import Function

my_function = Function(
    compute_job,
    name="my-median-function",
    maximum_concurrency=10,  # Limit concurrent runs of THIS function
    cpus=1.0,
    memory=2048,
    timeout=3600,
)
```

- **`maximum_concurrency`**: Limits parallel execution for this specific function
- Default is typically 10 if not specified
- Useful for controlling resource usage and preventing overwhelming the system

### Compute Resources
- **CPU allocation:** Configurable from 0.25 to 8+ CPUs per job
- **Memory allocation:** Configurable from 512 MB to 32+ GB per job
- **Timeout:** Maximum execution time per job (configurable, typically up to 3600 seconds/1 hour)
- **Return payload limit:** 1 MB maximum size for data returned from the function
  - **What this means:** The function's return value (as serialized JSON/data) cannot exceed 1 MB
  - **Why it exists:** Prevents platform abuse and ensures stability
  - **Workaround:** For large results (e.g., raster data), save to cloud storage and return a reference (URL/ID) instead
  - **Example:** Instead of returning a 50 MB image array, save it to storage and return `{"result_url": "s3://bucket/result.tif"}`

### Rate Limits
- **Compute seconds quota:** Aggregated across all queries
- **Monthly reset:** Usage quotas typically reset monthly
- **Rate limit errors:** When exceeded, API returns `RateLimitExceeded` error

## Current Implementation in earthone-medians

### Default Configuration
Our `ServerlessMedianComputer` uses these defaults:

```python
from earthone_medians import compute_sentinel2_median_serverless

result = compute_sentinel2_median_serverless(
    bbox=[...],
    start_date="...",
    end_date="...",
    cpus=1.0,        # Default: 1 CPU
    memory=2048,     # Default: 2 GB RAM
    # maximum_concurrency=10 (set in Function creation)
)
```

### How It Works in the Code
In `earthone_medians/serverless.py`, the Function is created with:

```python
func = Function(
    compute_median_job,
    name=f"earthone-medians-{sensor}-{timestamp}",
    image="python3.10:latest",
    cpus=cpus,                    # From user parameter
    memory=memory,                # From user parameter
    timeout=3600,                 # 1 hour
    maximum_concurrency=10,       # Hardcoded limit
    requirements=[...]
)
```

## Best Practices for Large-Scale Processing

### 1. Batch Processing Multiple Regions

If processing multiple regions, consider your concurrency strategy:

**Option A: Sequential Processing (Safe)**
```python
from earthone_medians import compute_sentinel2_median_serverless

regions = [
    {"name": "Region1", "bbox": [...]},
    {"name": "Region2", "bbox": [...]},
    # ... many more
]

for region in regions:
    result = compute_sentinel2_median_serverless(
        bbox=region['bbox'],
        start_date="2023-01-01",
        end_date="2023-12-31",
        cpus=2.0,
        memory=4096,
    )
    # Wait for each to complete before starting next
```

**Option B: Parallel Processing (Faster, Watch Limits)**
```python
import asyncio
from earthone_medians import compute_sentinel2_median_serverless

async def process_region(region):
    return compute_sentinel2_median_serverless(
        bbox=region['bbox'],
        start_date="2023-01-01",
        end_date="2023-12-31",
        cpus=1.0,
        memory=2048,
    )

# Process up to 10 regions in parallel
regions = [...]  # Your region list
batch_size = 10  # Stay well under 1,000 concurrent job limit

for i in range(0, len(regions), batch_size):
    batch = regions[i:i+batch_size]
    results = await asyncio.gather(*[process_region(r) for r in batch])
    # Process results, then continue to next batch
```

### 2. Resource Allocation Strategy

**Small areas (< 10km²):**
```python
cpus=0.5
memory=1024
```

**Medium areas (10-100km²):**
```python
cpus=1.0
memory=2048  # Default
```

**Large areas (100-1000km²):**
```python
cpus=2.0
memory=4096
```

**Very large areas (> 1000km²):**
```python
cpus=4.0
memory=8192
```

### 3. Handling Rate Limits

```python
import time
from earthone_medians import compute_sentinel2_median_serverless

def process_with_retry(bbox, max_retries=3):
    for attempt in range(max_retries):
        try:
            return compute_sentinel2_median_serverless(
                bbox=bbox,
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
        except Exception as e:
            if "RateLimitExceeded" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt * 60  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

### 4. Monitoring Job Status

The serverless implementation returns job information:

```python
result = compute_sentinel2_median_serverless(...)

print(f"Job ID: {result['job_id']}")
print(f"Status: {result['status']}")
print(f"Function: {result['function_name']}")

# Job logs are included in result
if 'logs' in result:
    print("Job Logs:", result['logs'])
```

## Checking Your Current Usage

To monitor your quota usage, you can:

1. **Check via EarthOne Dashboard:**
   - Log in to https://app.earthone.earthdaily.com
   - Navigate to Compute section
   - View current usage and quotas

2. **Query via API:**
   ```python
   from earthdaily.earthone import Auth
   
   auth = Auth()
   # API methods to check quotas (check latest docs for exact methods)
   ```

## What Happens When Limits Are Reached?

1. **Concurrent job limit (1,000):**
   - New jobs are queued or rejected
   - You'll receive an error indicating limit reached
   - Wait for running jobs to complete

2. **Rate limit (compute seconds):**
   - API returns `RateLimitExceeded` error
   - Implement exponential backoff and retry
   - Consider waiting until quota resets (monthly)

3. **Function-specific concurrency:**
   - Jobs queue until slots are available
   - Automatically handled by the platform

## Requesting Quota Increases

If your workflow requires higher limits:

1. Contact EarthOne Support: https://support.earthdaily.com
2. Provide details:
   - Current quota limits
   - Required quota increases
   - Use case description
   - Expected job volumes

## Summary Table

| Limit Type | Default Value | Configurable | Impact |
|------------|---------------|--------------|--------|
| **Concurrent invocations** | 1,000 per user | No (platform limit) | Total parallel jobs |
| **Function concurrency** | 10 (typical) | Yes (via `maximum_concurrency`) | Jobs per function |
| **CPU per job** | 1.0 | Yes (0.25-8+) | Job performance |
| **Memory per job** | 2048 MB | Yes (512-32768+ MB) | Job memory limit |
| **Timeout** | 3600s | Yes | Max job duration |
| **Return payload** | 1 MB | No | Result size limit (use storage for large data) |
| **Compute seconds** | Varies by account | No (contact support) | Monthly usage |

## References

- EarthOne Compute Documentation: https://docs.earthone.earthdaily.com/guides/compute.html
- EarthOne Quotas & Limits: https://docs.earthone.earthdaily.com/guides/quota.html
- EarthOne Support: https://support.earthdaily.com
