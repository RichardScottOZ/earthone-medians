"""Setup script for earthone-medians package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="earthone-medians",
    version="0.1.0",
    author="Richard Scott",
    description="Build median time series for Sentinel-2, Landsat, and ASTER using EarthOne EarthDaily API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RichardScottOZ/earthone-medians",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "earthdaily>=0.0.1",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "rasterio>=1.2.0",
        "geopandas>=0.10.0",
    ],
    entry_points={
        "console_scripts": [
            "earthone-medians=earthone_medians.cli:main",
        ],
    },
)
