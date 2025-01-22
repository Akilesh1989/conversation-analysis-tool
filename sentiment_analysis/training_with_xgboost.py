import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle



SENTIMENT_DATA_PATH = "sentiment_analysis"

ALGO = 'xgboost'

# Load the dataset
input_file = os.path.join('assignment_details', 'topical_chat_10000.csv')
df = pd.read_csv(input_file)

# Initialize sentence transformer model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Encode messages using SentenceTransformer
print("Encoding messages...")
X = model.encode(df['message'].tolist(), show_progress_bar=True)

# Convert sentiment labels to numerical values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    learning_rate=0.1,
    max_depth=6,
    n_estimators=100,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Get metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

# Print metrics
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Create filename with metrics
metrics_str = f"acc_{accuracy:.3f}"
for label, metrics in report.items():
    if label not in ['accuracy', 'macro avg', 'weighted avg']:
        metrics_str += f"_{label}_p{metrics['precision']:.2f}_r{metrics['recall']:.2f}_f1{metrics['f1-score']:.2f}"

model_filename = f'{ALGO}_{model_name}.pkl'


with open(os.path.join(SENTIMENT_DATA_PATH, model_filename), 'wb') as f:
    pickle.dump(xgb_model, f)
    
with open(SENTIMENT_DATA_PATH + f'/{model_filename.split(".")[0]}_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save sentence transformer model name for later use
with open(SENTIMENT_DATA_PATH + f'/{model_filename.split(".")[0]}_transformer_model_name.txt', 'w') as f:
    f.write('all-MiniLM-L6-v2')

print(f"Models and encoders saved successfully! Model saved as: {model_filename}")