FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy folders separately (Respecting your structure)
COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY .env ./

# 4. CRITICAL: Set PYTHONPATH so 'scripts' can see the 'app' folder
ENV PYTHONPATH=/app

EXPOSE 8000

# Default startup command (can be overridden by compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]