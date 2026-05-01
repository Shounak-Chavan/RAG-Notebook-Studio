FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

# Expose port 80
EXPOSE 80

# Run app
CMD ["python", "app.py"]