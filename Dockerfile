# Dockerfile for Render deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads 3dModels static

# Expose port (Render will override with $PORT)
EXPOSE 8000

# Start command
CMD ["uvicorn", "cloud_server:app", "--host", "0.0.0.0", "--port", "8000"]