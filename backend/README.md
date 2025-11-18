# MRMS Radar Backend

FastAPI backend that fetches and processes MRMS radar data.

## Quick Start

### Local Development

1. Install system dependencies (for eccodes):
   - **Ubuntu/Debian**: `sudo apt-get install libeccodes-dev`
   - **macOS**: `brew install eccodes`
   - **Windows**: Use WSL or Docker

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Run the server:
```bash
python app.py
# Or:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t mrms-radar-backend .
docker run -p 8000:8000 mrms-radar-backend
```

## API Endpoints

- `GET /` - Health check
- `GET /radar/latest/metadata` - Get radar metadata (bounds, timestamp, dBZ range)
- `GET /radar/latest/image` - Get radar image as PNG

## Caching

The backend caches processed radar data for 3 minutes to reduce load on MRMS servers and improve response times.

## Troubleshooting

### eccodes/cfgrib installation issues

If you encounter errors with `cfgrib`, ensure `eccodes` is installed on your system. On some platforms, you may need to:

1. Install eccodes from source
2. Set environment variables pointing to the installation
3. Use Docker (recommended for consistent environment)

### GRIB2 parsing errors

If the GRIB2 file structure differs from expected, check:
- The variable names in the file (may need to adjust in `parse_grib2`)
- Coordinate system (may need to adjust bounds calculation)

