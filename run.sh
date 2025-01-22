#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p q_and_a/FAISS_MODELS summarization/output

# Train models and generate data
echo "Training sentiment analysis models and generating model file."
python sentiment_analysis/training_with_xgboost.py

echo "Generating summarization data..."
python summarization/main.py

echo "Building Q and A model"
python q_and_a/build.py

streamlit run app.py