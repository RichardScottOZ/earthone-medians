# EarthOne Medians - Two Approaches

This package provides **TWO** distinct approaches for computing temporal median composites, each optimized for different use cases.

## Approach 1: Workbench/Interactive (Dynamic Compute)

**Best for:**
- Interactive analysis in Jupyter notebooks
- Exploration and visualization in EarthOne Workbench
- Quick prototyping
- Small to medium-sized areas

**Uses:** `earthdaily.earthone.dynamic_compute.Mosaic`

### Python API

```python
from earthone_medians import compute_sentinel2_median_workbench

# Returns a Mosaic object that can be visualized
result = compute_sentinel2_median_workbench(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],
    resolution=10,
)

# In a Jupyter notebook, visualize the mosaic:
# from earthdaily.earthone import dynamic_compute as dc
# m = dc.map
# result['mosaic'].visualize("Median", m)
```

### Characteristics
- **Fast** - Creates mosaic layer immediately
- **Interactive** - Can visualize and explore results in real-time
- **Memory-efficient** - Computation happens on-demand
- **Ideal for notebooks** - Works seamlessly in Jupyter/Workbench

## Approach 2: Serverless Compute (Batch Processing)

**Best for:**
- Large-scale batch processing
- Production workflows
- Processing many areas
- Heavy computation workloads

**Uses:** `earthdaily.earthone.compute.Function`

### Python API

```python
from earthone_medians import compute_sentinel2_median_serverless

# Submits a job to EarthOne compute infrastructure
result = compute_sentinel2_median_serverless(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B2", "B3", "B4", "B8"],
    resolution=10,
    cpus=2.0,  # Configure compute resources
    memory=4096,  # MB
)

# Returns job results
print(f"Job ID: {result['job_id']}")
print(f"Status: {result['status']}")
print(f"Result: {result['result']}")
```

### Characteristics
- **Scalable** - Runs on EarthOne's cloud infrastructure
- **Configurable** - Control CPU, memory allocation
- **Batch processing** - Process multiple jobs in parallel
- **Production-ready** - Reliable, logged, monitored
- **Platform limits** - Max 1,000 concurrent jobs per user (see [SERVERLESS_LIMITS.md](SERVERLESS_LIMITS.md))

## Comparison Table

| Feature | Workbench/Interactive | Serverless Compute |
|---------|----------------------|-------------------|
| **Use Case** | Exploration, notebooks | Production, batch |
| **Execution** | On-demand, interactive | Batch job submission |
| **Resources** | Dynamic | Configurable (CPU/memory) |
| **Results** | Mosaic object | Job results + logs |
| **Visualization** | Built-in | Requires download |
| **Scale** | Small-medium | Large-scale |
| **Speed** | Immediate mosaic creation | Job queue + processing |

## CLI Support

The CLI currently uses the workbench approach by default:

```bash
# Uses workbench approach
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31"
```

To use serverless compute from CLI, you can use the `--method` flag:

```bash
# Uses serverless compute
earthone-medians sentinel2 \
    --bbox "115.0,-32.0,116.0,-31.0" \
    --start-date "2023-01-01" \
    --end-date "2023-12-31" \
    --method serverless \
    --cpus 2.0 \
    --memory 4096
```

## Backward Compatibility

For backward compatibility, the default functions use the workbench approach:

```python
from earthone_medians import compute_sentinel2_median

# This uses workbench approach by default
result = compute_sentinel2_median(...)
```

## Complete Examples

### Example 1: Interactive Exploration (Workbench)

```python
import earthdaily.earthone.dynamic_compute as dc
from earthone_medians import compute_sentinel2_median_workbench

# Create a map
m = dc.map
m.center = (-31.5, 115.5)
m.zoom = 8

# Compute and visualize median
result = compute_sentinel2_median_workbench(
    bbox=[115.0, -32.0, 116.0, -31.0],
    start_date="2023-01-01",
    end_date="2023-12-31",
    bands=["B4", "B3", "B2"],  # True color
)

# Visualize on map
result['mosaic'].visualize("S2 Median 2023", m)
```

### Example 2: Production Batch Processing (Serverless)

```python
from earthone_medians import compute_sentinel2_median_serverless

# Process multiple regions
regions = [
    {"name": "Perth", "bbox": [115.0, -32.0, 116.0, -31.0]},
    {"name": "Sydney", "bbox": [150.0, -34.0, 151.0, -33.0]},
    {"name": "Brisbane", "bbox": [152.0, -28.0, 153.0, -27.0]},
]

jobs = []
for region in regions:
    print(f"Processing {region['name']}...")
    result = compute_sentinel2_median_serverless(
        bbox=region['bbox'],
        start_date="2023-01-01",
        end_date="2023-12-31",
        cpus=2.0,
        memory=4096,
    )
    jobs.append({"region": region['name'], "job_id": result['job_id']})
    print(f"  Job submitted: {result['job_id']}")

print("\nAll jobs submitted:")
for job in jobs:
    print(f"  {job['region']}: {job['job_id']}")
```

## Choosing the Right Approach

**Use Workbench/Interactive when:**
- Working in Jupyter notebooks
- Exploring data interactively
- Need immediate visualization
- Processing small to medium areas
- Prototyping and development

**Use Serverless Compute when:**
- Running production workflows
- Processing large areas or many regions
- Need reproducible batch processing
- Want to control compute resources
- Building automated pipelines

## Authentication

Both approaches use the same EarthOne authentication:

```bash
# Interactive login
earthone auth login

# Or environment variables
export EARTHONE_CLIENT_ID="your-client-id"
export EARTHONE_CLIENT_SECRET="your-client-secret"
```
