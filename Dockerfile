# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install ONLY what is absolutely necessary for the OS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libtesseract-dev \
    tesseract-ocr \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install dependencies (This is the slow part, done only once)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Copy your code (This is the fast part)
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
EXPOSE 8501

# Default start command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]