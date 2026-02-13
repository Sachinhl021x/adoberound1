# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies (e.g., for OpenCV or build tools if needed)
# tesseract-ocr is needed for OCR if we enabled it, let's include it to be safe for "production"
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Make scripts executable
RUN chmod +x scripts/start.sh scripts/serve.py

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Run the start script
CMD ["./scripts/start.sh"]
