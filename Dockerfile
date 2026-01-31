# 1. Upgrade to 3.11 for faster dictionary lookups and ML performance
FROM python:3.11-slim

WORKDIR /app

# 2. Keep your system dependencies + add libgomp1 for XGBoost
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Leverage caching for dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy code and artifacts
COPY . .

# 5. Document the Bidding API port
EXPOSE 8000

# 6. Default to running the FastAPI app (Day 18 setup)
# This assumes your FastAPI instance is named 'app' inside 'src/main.py'
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]