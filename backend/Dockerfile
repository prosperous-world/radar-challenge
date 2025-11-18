FROM python:3.10.8-slim

WORKDIR /app

# Install system dependencies for eccodes (required by cfgrib)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create cache directory
RUN mkdir -p /tmp/radar_cache

# Set environment variables to suppress ECCODES warnings
ENV ECCODES_DEBUG=0
ENV ECCODES_LOG_STREAM=/dev/null
ENV GRIB_API_LOG_STREAM=/dev/null

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

