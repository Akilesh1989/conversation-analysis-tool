# Use Python 3.9 base image
FROM python:3.9-slim

RUN python --version

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install and setup Python environment
RUN pip install uv
RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"
RUN . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy only the necessary files and directories
COPY . .

# Create necessary directories
RUN mkdir -p q_and_a/FAISS_MODELS summarization/output

# Set environment variables
ENV PYTHONPATH=/app
ENV NLTK_DATA=/app/nltk_data

RUN pip install -e .

# Expose Streamlit port
EXPOSE 8501

# Set the entrypoint command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
