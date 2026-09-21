# Use a lightweight Python base image
FROM python:3.11-slim

# Install system dependencies for OCR and PDF handling
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run your backend (Flask app)
# CMD ["python", "app.py"]

# Run backend with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
