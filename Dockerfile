# Use Python 3.12 slim — stable, smaller image, faster cold starts on Cloud Run
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependencies first so Docker caches this layer
# Only re-runs pip install if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Cloud Run expects the app to listen on port 8080
EXPOSE 8080

# Run the MCP server
CMD ["python", "main.py"]
