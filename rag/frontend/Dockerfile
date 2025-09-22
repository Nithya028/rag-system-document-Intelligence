# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your frontend code
COPY . .

# Expose Streamlit default port
EXPOSE 8800

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8800
ENV STREAMLIT_SERVER_HEADLESS=true
ENV BACKEND_URL=http://backend:8002

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py","--server.port=8800", "--server.address=0.0.0.0"]
