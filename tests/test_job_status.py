"""Check job status attributes."""
from earthdaily.earthone.compute import Job

# List recent jobs
jobs = list(Job.list())
if jobs:
    job = jobs[0]
    print(f"Job ID: {job.id}")
    print(f"  status: {job.status}")
    print(f"  state: {job.state}")
    print(f"  exit_code: {job.exit_code}")
    print(f"  error_reason: {job.error_reason}")
else:
    print("No jobs found")
