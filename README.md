# MRMS Radar Visualization

A real-time radar visualization application that fetches and displays MRMS (Multi-Radar Multi-Sensor) Reflectivity at Lowest Altitude (RALA) data from NOAA.

## Features

- **Real-time Data**: Fetches latest radar data directly from NOAA MRMS servers
- **Interactive Map**: React Leaflet map with radar overlay
- **Auto-refresh**: Automatically updates every 3 minutes
- **Color-coded Reflectivity**: Classic radar colormap (green → yellow → orange → red → purple)
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

### Backend (FastAPI)
- Fetches GRIB2 data from `mrms.ncep.noaa.gov`
- Parses GRIB2 files using `cfgrib` and `xarray`
- Converts reflectivity values to colored PNG images
- Serves metadata and images via REST API
- Implements caching (3-minute cache duration)

### Frontend (React + Vite)
- React Leaflet for map visualization
- Auto-refresh functionality
- Clean, modern UI with radar legend

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: `cfgrib` requires `eccodes` library. On Linux/Mac, install it via:
- Ubuntu/Debian: `sudo apt-get install libeccodes-dev`
- macOS: `brew install eccodes`
- Windows: Use WSL or Docker (recommended)

4. Run the backend:
```bash
python app.py
# Or with uvicorn directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (or copy `.env.example`):
```bash
cp .env.example .env
```

4. Update `.env` with your backend URL:
```
VITE_API_URL=http://localhost:8000
```

5. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Deployment

### Backend Deployment (Render/Railway/Fly)

#### Option 1: Docker (Recommended)

1. Build and push your Docker image, or use Render's Docker deployment
2. Ensure the container has internet access to fetch MRMS data
3. Set environment variables if needed

#### Option 2: Python Buildpack

1. Connect your repository to Render
2. Select "Web Service"
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add buildpack/system dependencies for `eccodes` if needed

### Frontend Deployment (Vercel/Netlify/Render Static)

1. Build the frontend:
```bash
npm run build
```

2. Deploy the `dist` folder to your static hosting service

3. Set environment variable `VITE_API_URL` to your deployed backend URL

## API Endpoints

### `GET /radar/latest/metadata`
Returns metadata about the latest radar image:
```json
{
  "north": 54.9,
  "south": 19.9,
  "east": -60.0,
  "west": -130.0,
  "timestamp": "2025-01-17T19:24:00Z",
  "min_dbz": -10.0,
  "max_dbz": 65.5
}
```

### `GET /radar/latest/image`
Returns the latest radar image as PNG.

## Data Source

- **Source**: NOAA MRMS (Multi-Radar Multi-Sensor)
- **Product**: Reflectivity at Lowest Altitude (RALA)
- **URL**: `https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/`
- **Update Frequency**: ~Every 2 minutes
- **Format**: GRIB2
- **Coverage**: CONUS (Continental United States)

## Technologies Used

### Backend
- **FastAPI**: Modern Python web framework
- **cfgrib + xarray**: GRIB2 file parsing
- **Pillow**: Image processing and PNG generation
- **NumPy**: Numerical operations
- **requests**: HTTP client for fetching MRMS data

### Frontend
- **React**: UI framework
- **Vite**: Build tool and dev server
- **React Leaflet**: Map component library
- **Leaflet**: Open-source mapping library

## License

This project is for educational/demonstration purposes. MRMS data is provided by NOAA and is in the public domain.

