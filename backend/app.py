"""
FastAPI backend for MRMS Radar visualization.
Fetches RALA data from NOAA, processes GRIB2 files, and serves radar images.
"""
import gc
import io
import gzip
import os
import time
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import requests
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import xarray as xr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress ECCODES warnings by setting environment variables
# These must be set before importing/using cfgrib
os.environ['ECCODES_DEBUG'] = '0'  # Disable ECCODES debug output
os.environ['ECCODES_LOG_STREAM'] = '/dev/null'  # Redirect ECCODES log to /dev/null
# Alternative: suppress all ECCODES messages
os.environ['GRIB_API_LOG_STREAM'] = '/dev/null'

app = FastAPI(title="MRMS Radar API")

# Get allowed origins from environment variable
# Expected format: "http://localhost:3000,http://localhost:8080,https://example.com"
ALLOW_ORIGINS_ENV = os.getenv("ALLOW_ORIGINS", "*")
if ALLOW_ORIGINS_ENV == "*":
    allow_origins = ["*"]
else:
    # Split comma-separated origins and strip whitespace
    allow_origins = [origin.strip() for origin in ALLOW_ORIGINS_ENV.split(",") if origin.strip()]

# CORS middleware to allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MRMS RALA data URL
MRMS_RALA_URL = "https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/MRMS_ReflectivityAtLowestAltitude.latest.grib2.gz"

# Cache settings
CACHE_DIR = Path("/tmp/radar_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DURATION_SECONDS = 180  # 3 minutes

# Radar color scale parameters
MIN_DBZ = -10
MAX_DBZ = 70

# Memory optimization: downsample factor (1 = no downsampling, 2 = half resolution, etc.)
DOWNSAMPLE_FACTOR = 2  # Reduce memory usage by downsampling data


class RadarProcessor:
    """Handles fetching, parsing, and rendering MRMS radar data."""
    
    def __init__(self):
        self.last_fetch_time: Optional[float] = None
        self.cached_metadata: Optional[Dict[str, Any]] = None
        self.cached_image_path: Optional[Path] = None
    
    def fetch_grib_data(self) -> bytes:
        """Fetch the latest GRIB2 file from MRMS."""
        try:
            response = requests.get(MRMS_RALA_URL, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to fetch MRMS data: {str(e)}"
            )
    
    def decompress_grib(self, gzipped_data: bytes) -> bytes:
        """Decompress gzipped GRIB2 data."""
        buf = io.BytesIO(gzipped_data)
        with gzip.GzipFile(fileobj=buf) as f:
            return f.read()
    
    def parse_grib2(self, grib_bytes: bytes) -> tuple:
        """
        Parse GRIB2 file and extract reflectivity data and metadata.
        Returns: (values_2d, lats, lons, timestamp)
        Memory optimized: uses context manager and downsampling.
        """
        # Write to temporary file for cfgrib
        temp_grib = CACHE_DIR / "temp_rala.grib2"
        temp_grib.write_bytes(grib_bytes)
        
        try:
            # Suppress ECCODES warnings (time truncation warnings are harmless)
            # ECCODES writes directly to file descriptor 2, so we need to redirect at FD level
            import sys
            
            # Save original stderr file descriptor
            stderr_fd = sys.stderr.fileno()
            stdout_fd = sys.stdout.fileno()
            
            # Open /dev/null for writing
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            
            try:
                # Save original stderr/stdout file descriptors
                saved_stderr = os.dup(stderr_fd)
                saved_stdout = os.dup(stdout_fd)
                
                # Redirect file descriptors 1 (stdout) and 2 (stderr) to /dev/null
                os.dup2(devnull_fd, stderr_fd)
                os.dup2(devnull_fd, stdout_fd)
                
                try:
                    # Use context manager to ensure dataset is closed
                    with xr.open_dataset(str(temp_grib), engine="cfgrib") as ds:
                        # Find the reflectivity variable (name may vary)
                        data_vars = list(ds.data_vars)
                        if not data_vars:
                            raise ValueError("No data variables found in GRIB file")
                        
                        # Try common variable names
                        var_name = None
                        for name in ['ReflectivityAtLowestAltitude', 'unknown', 'r', 'ref']:
                            if name in data_vars:
                                var_name = name
                                break
                        
                        if var_name is None:
                            var_name = data_vars[0]
                        
                        data_array = ds[var_name]
                        
                        # Downsample to reduce memory usage
                        if DOWNSAMPLE_FACTOR > 1:
                            # Use xarray's isel to downsample
                            data_array = data_array.isel(
                                {dim: slice(None, None, DOWNSAMPLE_FACTOR) 
                                 for dim in data_array.dims}
                            )
                        
                        # Extract values (load into memory)
                        values = data_array.values.copy()  # Copy to ensure data is in memory
                        
                        # Get coordinates
                        if 'latitude' in ds.coords and 'longitude' in ds.coords:
                            lats = ds.coords['latitude'].values
                            lons = ds.coords['longitude'].values
                            
                            # Downsample coordinates if we downsampled data
                            if DOWNSAMPLE_FACTOR > 1:
                                if lats.ndim == 1:
                                    lats = lats[::DOWNSAMPLE_FACTOR]
                                if lons.ndim == 1:
                                    lons = lons[::DOWNSAMPLE_FACTOR]
                            
                            # If 1D, create meshgrid
                            if lats.ndim == 1 and lons.ndim == 1:
                                lons_2d, lats_2d = np.meshgrid(lons, lats)
                            else:
                                # Downsample 2D coordinates
                                if DOWNSAMPLE_FACTOR > 1:
                                    lats_2d = lats[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
                                    lons_2d = lons[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
                                else:
                                    lats_2d = lats
                                    lons_2d = lons
                        else:
                            # Fallback: try to get from attributes or use defaults
                            # MRMS CONUS grid is roughly 19.9 to 54.9 N, -130 to -60 W
                            nrows, ncols = values.shape
                            lats_2d = np.linspace(19.9, 54.9, nrows)
                            lons_2d = np.linspace(-130.0, -60.0, ncols)
                            lons_2d, lats_2d = np.meshgrid(lons_2d, lats_2d)
                        
                        # Get timestamp from attributes or use current time
                        timestamp = None
                        if 'time' in ds.coords:
                            time_val = ds.coords['time'].values
                            if hasattr(time_val, 'item'):
                                timestamp = time_val.item()
                            else:
                                timestamp = str(time_val)
                        
                        if timestamp is None:
                            # Try to get from file attributes
                            timestamp = datetime.now(timezone.utc).isoformat()
                        
                        # Dataset will be automatically closed by context manager
                        return values, lats_2d, lons_2d, timestamp
                finally:
                    # Restore original file descriptors
                    os.dup2(saved_stderr, stderr_fd)
                    os.dup2(saved_stdout, stdout_fd)
                    os.close(saved_stderr)
                    os.close(saved_stdout)
            finally:
                # Close /dev/null file descriptor
                os.close(devnull_fd)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse GRIB2 file: {str(e)}"
            )
        finally:
            # Clean up temp file
            if temp_grib.exists():
                temp_grib.unlink()
            # Force garbage collection
            gc.collect()
    
    def dbz_to_rgba(self, norm_val: float) -> tuple:
        """
        Convert normalized dBZ value (0-1) to RGBA color.
        Classic radar colormap: green -> yellow -> orange -> red -> purple
        """
        if norm_val < 0.2:
            # Light green for light precipitation
            return (0, 200, 0, 120)
        elif norm_val < 0.4:
            # Green to yellow
            t = (norm_val - 0.2) / 0.2
            return (int(255 * t), 200, 0, 150)
        elif norm_val < 0.6:
            # Yellow to orange
            t = (norm_val - 0.4) / 0.2
            return (255, int(200 * (1 - t)), 0, 180)
        elif norm_val < 0.8:
            # Orange to red
            t = (norm_val - 0.6) / 0.2
            return (255, int(100 * (1 - t)), 0, 200)
        else:
            # Red to purple for heavy precipitation
            t = (norm_val - 0.8) / 0.2
            return (int(255 * (1 - t * 0.5)), 0, int(255 * t * 0.5), 220)
    
    def create_radar_image(self, values: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> Image.Image:
        """
        Convert reflectivity values to a colored PNG image.
        Uses vectorized operations for better performance.
        Memory optimized: processes in-place where possible.
        """
        # Clip and normalize values
        clipped = np.clip(values, MIN_DBZ, MAX_DBZ)
        # Handle NaN values (in-place to save memory)
        np.nan_to_num(clipped, nan=MIN_DBZ, copy=False)
        norm = (clipped - MIN_DBZ) / (MAX_DBZ - MIN_DBZ)
        
        # Create RGBA array
        height, width = values.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Vectorized color mapping
        # Light green (0-0.2)
        mask1 = norm < 0.2
        rgba[mask1] = [0, 200, 0, 120]
        
        # Green to yellow (0.2-0.4)
        mask2 = (norm >= 0.2) & (norm < 0.4)
        if np.any(mask2):
            t2 = (norm[mask2] - 0.2) / 0.2
            rgba[mask2] = np.column_stack([
                (255 * t2).astype(np.uint8),
                np.full(len(t2), 200, dtype=np.uint8),
                np.zeros(len(t2), dtype=np.uint8),
                np.full(len(t2), 150, dtype=np.uint8)
            ])
        
        # Yellow to orange (0.4-0.6)
        mask3 = (norm >= 0.4) & (norm < 0.6)
        if np.any(mask3):
            t3 = (norm[mask3] - 0.4) / 0.2
            rgba[mask3] = np.column_stack([
                np.full(len(t3), 255, dtype=np.uint8),
                (200 * (1 - t3)).astype(np.uint8),
                np.zeros(len(t3), dtype=np.uint8),
                np.full(len(t3), 180, dtype=np.uint8)
            ])
        
        # Orange to red (0.6-0.8)
        mask4 = (norm >= 0.6) & (norm < 0.8)
        if np.any(mask4):
            t4 = (norm[mask4] - 0.6) / 0.2
            rgba[mask4] = np.column_stack([
                np.full(len(t4), 255, dtype=np.uint8),
                (100 * (1 - t4)).astype(np.uint8),
                np.zeros(len(t4), dtype=np.uint8),
                np.full(len(t4), 200, dtype=np.uint8)
            ])
        
        # Red to purple (0.8-1.0)
        mask5 = norm >= 0.8
        if np.any(mask5):
            t5 = (norm[mask5] - 0.8) / 0.2
            rgba[mask5] = np.column_stack([
                (255 * (1 - t5 * 0.5)).astype(np.uint8),
                np.zeros(len(t5), dtype=np.uint8),
                (255 * t5 * 0.5).astype(np.uint8),
                np.full(len(t5), 220, dtype=np.uint8)
            ])
        
        # Create PIL Image
        img = Image.fromarray(rgba, mode='RGBA')
        
        # Clean up large arrays
        del rgba, norm, clipped, mask1, mask2, mask3, mask4, mask5
        gc.collect()
        
        return img
    
    def process_radar_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Main processing function: fetch, parse, and render radar data.
        Returns metadata dict with bounds, timestamp, and image path.
        Memory optimized: cleans up large objects immediately after use.
        """
        current_time = time.time()
        
        # Check cache
        if not force_refresh and self.last_fetch_time is not None:
            age = current_time - self.last_fetch_time
            if age < CACHE_DURATION_SECONDS and self.cached_metadata is not None:
                return self.cached_metadata
        
        # Fetch new data
        print("Fetching latest MRMS data...")
        gzipped_data = self.fetch_grib_data()
        grib_bytes = self.decompress_grib(gzipped_data)
        
        # Clean up compressed data immediately
        del gzipped_data
        gc.collect()
        
        # Parse GRIB2
        print("Parsing GRIB2 file...")
        values, lats, lons, timestamp = self.parse_grib2(grib_bytes)
        
        # Clean up GRIB bytes immediately
        del grib_bytes
        gc.collect()
        
        # Compute bounds before creating image
        north = float(np.nanmax(lats))
        south = float(np.nanmin(lats))
        east = float(np.nanmax(lons))
        west = float(np.nanmin(lons))
        min_dbz = float(np.nanmin(values))
        max_dbz = float(np.nanmax(values))
        
        # Create image
        print("Rendering radar image...")
        img = self.create_radar_image(values, lats, lons)
        
        # Clean up large arrays immediately
        del values, lats, lons
        gc.collect()
        
        # Save image
        image_path = CACHE_DIR / "rala_latest.png"
        img.save(image_path, format='PNG', optimize=True)
        
        # Clean up image object
        del img
        gc.collect()
        
        # Update cache
        self.last_fetch_time = current_time
        self.cached_metadata = {
            "north": north,
            "south": south,
            "east": east,
            "west": west,
            "timestamp": str(timestamp),
            "min_dbz": min_dbz,
            "max_dbz": max_dbz,
        }
        self.cached_image_path = image_path
        
        return self.cached_metadata


# Global processor instance
processor = RadarProcessor()


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "MRMS Radar API"}


@app.get("/radar/latest/metadata")
def get_metadata():
    """
    Get metadata for the latest radar image.
    Returns bounds, timestamp, and reflectivity range.
    """
    try:
        metadata = processor.process_radar_data()
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/radar/latest/image")
def get_image():
    """
    Get the latest radar image as PNG.
    """
    try:
        # Ensure data is processed
        processor.process_radar_data()
        
        if processor.cached_image_path is None or not processor.cached_image_path.exists():
            raise HTTPException(status_code=404, detail="Radar image not available")
        
        image_bytes = processor.cached_image_path.read_bytes()
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

