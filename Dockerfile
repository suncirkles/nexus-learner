FROM python:3.11-slim

# Install system dependencies (required for Tesseract OCR / PyMuPDF)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Environment variable defaults
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
EXPOSE 8501

# The command will be overridden by docker-compose for UI vs API
CMD ["python", "app.py"]
