# Use Python 3.9 base image
FROM python:3.9-slim

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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy only the necessary files and directories
COPY assignment_details /app/assignment_details
COPY q_and_a /app/q_and_a

COPY sentiment_analysis/analyse.py /app/sentiment_analysis/analyse.py
COPY sentiment_analysis/predict.py /app/sentiment_analysis/predict.py
COPY sentiment_analysis/train.py /app/sentiment_analysis/train.py

COPY summarization/main.py /app/summarization/main.py
COPY summarization/summarizer.py /app/summarization/summarizer.py
COPY summarization/text_summaries.txt /app/summarization/text_summaries.txt


COPY app.py /app/app.py
COPY setup.py /app/setup.py

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
