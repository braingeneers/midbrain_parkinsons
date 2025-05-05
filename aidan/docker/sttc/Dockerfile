# Use Python 3.11 base image
FROM python:3.11-slim

# Set environment variable to prevent interactive prompts during package installs.
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages (in this example, installing "time")
RUN apt-get update && \
    apt-get install -y time && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the current directory (docker/sttc) into the container.
COPY docker/sttc/requirements.txt /app/requirements.txt

# Set the working directory.
WORKDIR /app

# Upgrade pip and install Python dependencies from requirements.txt.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container.
COPY . /app

# Set up an environment variable for PRP setup (if required).
ENV ENDPOINT_URL="https://s3.braingeneers.gi.ucsc.edu"

# Set the default command to run your generate_sttc.py script.
CMD ["python", "generate-plots/generate_okun.py"]
