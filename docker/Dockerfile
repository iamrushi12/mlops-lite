# Use official Python image
FROM python:3.10-slim

# Set work directory inside the container
WORKDIR /app

# Copy only requirements first (for layer caching)
COPY requirements.txt .

# Install required Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy entire project into container
COPY . .

# Set default entrypoint
CMD ["python", "src/pipeline/sagemaker_pipeline.py"]
