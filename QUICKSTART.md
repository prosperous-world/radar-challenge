# Quick Start Guide

Get the MRMS Radar application running in minutes!

## Prerequisites

- **Python 3.10.8+** (for backend)
- **Node.js 18+** (for frontend)
- **eccodes library** (for GRIB2 parsing - see installation below)

## Step 1: Install eccodes (Required for Backend)

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install libeccodes-dev
```

### macOS
```bash
brew install eccodes
```

### Windows
- Use **WSL** (Windows Subsystem for Linux) - recommended
- Or use **Docker** (see Dockerfile in backend/)
- Or install eccodes manually from [ECMWF](https://confluence.ecmwf.int/display/ECC)

## Step 2: Start Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Backend will be available at: `http://localhost:8000`

Test it: Open `http://localhost:8000/radar/latest/metadata` in your browser.

## Step 3: Start Frontend

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Start dev server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## Step 4: View the Radar!

Open `http://localhost:3000` in your browser. You should see:
- A map of the United States
- Radar overlay showing precipitation
- Color-coded legend
- Auto-refresh every 3 minutes

## Troubleshooting

### Backend won't start - eccodes error
- Make sure eccodes is installed (see Step 1)
- On Linux, you may need: `sudo apt-get install libeccodes-dev gcc g++`
- Try using Docker instead: `cd backend && docker build -t radar . && docker run -p 8000:8000 radar`

### Frontend shows "Failed to fetch"
- Make sure backend is running on port 8000
- Check that `VITE_API_URL` in `.env` matches your backend URL
- Check browser console for CORS errors (backend should handle this)

### No radar data showing
- Check backend logs for errors
- Verify you can access `https://mrms.ncep.noaa.gov/2D/ReflectivityAtLowestAltitude/` in your browser
- The first fetch may take 10-30 seconds (downloading and processing GRIB2 file)

## Next Steps

- Deploy to production (see README.md)
- Customize colors and styling
- Add more radar products
- Implement user preferences

