FROM python:3.11-slim

WORKDIR /app

# xgboost's shared library needs libgomp (OpenMP) at runtime -- python:3.11-slim
# doesn't ship it by default, unlike the macOS dev environment where libomp comes
# from Homebrew.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models_artifacts/ ./models_artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
