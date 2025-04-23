# Use an official Python runtime as a parent image
# Using slim-buster for a smaller image size
FROM python:3.10-slim-buster

# Set environment variables
# Prevents Python from writing pyc files to disc (optional)
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to terminal (useful for logging)
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by libraries like OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Add other dependencies if needed by your libraries
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install project dependencies
# Separate torch installation can sometimes help with compatibility
# Using CPU-only versions here for broader compatibility initially.
# For GPU support, you'd need nvidia base images and specific torch cuda versions.
RUN pip install --no-cache-dir -r requirements.txt \
    # Explicitly install CPU version of torch (adjust version as needed)
    torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
    # Install diffusers and transformers (might be in requirements, but ensures they are present)
    # Consider pinning versions based on compatibility with torch version
    # diffusers transformers accelerate

# Copy the source code and scripts into the container
COPY src/ /app/src
COPY scripts/ /app/scripts

# Make scripts executable (if needed, e.g., for entrypoint)
# RUN chmod +x /app/scripts/generate.py

# Define the entrypoint for the container.
# This makes the container execute your script when it starts.
ENTRYPOINT ["python", "/app/scripts/generate.py"]

# Define default command (optional, can be overridden)
# Example: CMD ["--help"] 