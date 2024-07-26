# Stage 1: Base Image with common dependencies
FROM python:3 AS base

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Stage 2: Streamlit Image
FROM base AS streamlit

# Run Streamlit
CMD ["streamlit", "run", "app.py"]

# Stage 3: FastAPI Image
FROM base AS fastapi

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
