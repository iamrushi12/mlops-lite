# Stage 1: Build
FROM python:3.12-slim as base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Default command
CMD ["python", "src/model/train.py"]
