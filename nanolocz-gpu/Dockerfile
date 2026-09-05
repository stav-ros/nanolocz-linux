# NanoLocz CPU-only Docker Image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libgl1-mesa-glx \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy package files
COPY pyproject.toml README.md NOTICE.md ./
COPY nanolocz/ ./nanolocz/
COPY tests/ ./tests/
COPY examples/ ./examples/

# Install Python package
RUN pip install --no-cache-dir -e ".[test]"

# Create data and results directories
RUN mkdir -p /data /results

# Set entrypoint
ENTRYPOINT ["nanolocz"]
CMD ["--help"]
