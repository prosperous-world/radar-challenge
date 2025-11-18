# MRMS Radar Frontend

React + Vite frontend for visualizing MRMS radar data.

## Quick Start

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Update `.env` with your backend URL:
```
VITE_API_URL=http://localhost:8000
```

4. Run development server:
```bash
npm run dev
```

5. Build for production:
```bash
npm run build
```

## Features

- Interactive Leaflet map
- Real-time radar overlay
- Auto-refresh every 3 minutes
- Color-coded reflectivity legend
- Responsive design

## Environment Variables

- `VITE_API_URL`: Backend API URL (default: `http://localhost:8000`)

