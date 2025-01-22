import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pycaret.classification import *
import os
import json

print("Loading data and model...")

SENTIMENT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'presentation', 'sentiment_analysis')


# Load data
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                       'sample_topical_chat.csv')
df = pd.read_csv(csv_path)

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")

# Generate embeddings
embeddings = model.encode(df['message'].tolist(), show_progress_bar=True)

# Create DataFrame with embeddings
embed_df = pd.DataFrame(embeddings)
embed_df['sentiment'] = df['sentiment']

print("Setting up PyCaret environment...")

# Initialize PyCaret setup with updated parameters
try:
    clf = setup(
        data=embed_df,
        target='sentiment',
        verbose=True,    # Enable verbose output
        html=True,       # Enable HTML output
        session_id=42    # For reproducibility
    )
    
    print("Training models...")
    
    # Compare models
    best_model = compare_models(n_select=1)
    
    # Finalize model
    final_model = finalize_model(best_model)
    
    # Save model
    save_model(final_model, f'{SENTIMENT_DATA_PATH}/sentiment_model')
    
    # Get model metrics
    metrics = pull()
    
    # Create report data
    report_data = {
        "model_performance": {
            "accuracy": float(metrics.loc[metrics.index[0], 'Accuracy']),
            "f1_score": float(metrics.loc[metrics.index[0], 'F1']),
            "precision": float(metrics.loc[metrics.index[0], 'Precision']),
            "recall": float(metrics.loc[metrics.index[0], 'Recall'])
        },
        "class_distribution": df['sentiment'].value_counts().to_dict(),
        "confusion_matrix": get_confusion_matrix(normalize=True).tolist()
    }
    
    # Save report
    report_path = os.path.join(SENTIMENT_DATA_PATH, "report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
    
    print("Training completed and report saved successfully!")
    
except Exception as e:
    print(f"Error during training: {str(e)}")