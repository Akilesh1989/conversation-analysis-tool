import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np

class SentimentPredictor:
    def __init__(self, model_path=None):
        """Initialize the predictor with optional model path"""
        self.current_dir = os.path.dirname(__file__)
        
        if model_path is None:
            model_path = os.path.join(self.current_dir, 'xgboost_all-MiniLM-L6-v2.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
        except Exception as e:
            raise Exception(f"Error initializing predictor: {str(e)}")

    def predict(self, text: str) -> tuple:
        """
        Predict sentiment for given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            tuple: (predicted_sentiment, confidence_score)
        """
        try:
            text_embedding = self.transformer.encode([text])
            probabilities = self.model.predict_proba(text_embedding)
            
            prediction = np.argmax(probabilities, axis=1)[0]
            confidence = np.max(probabilities)
            
            sentiment_map = {
                0: "Angry",
                1: "Curious to dive deeper",
                2: "Disgusted",
                3: "Fearful",
                4: "Happy",
                5: "Neutral",
                6: "Sad",
                7: "Surprised"
            }
            
            predicted_sentiment = sentiment_map.get(prediction, "Unknown")
            return predicted_sentiment, confidence
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return "Error", 0.0

def predict_single_text(text: str, model_path=None) -> tuple:
    """
    Wrapper function for sentiment prediction
    
    Args:
        text: Input text to analyze
        model_path: Optional path to model file
        
    Returns:
        tuple: (predicted_sentiment, confidence_score)
    """
    try:
        predictor = SentimentPredictor(model_path)
        return predictor.predict(text)
    except Exception as e:
        print(f"Error in sentiment prediction: {str(e)}")
        return "Error", 0.0

if __name__ == "__main__":
    text = input("Enter text to analyze sentiment: ")
    sentiment, confidence = predict_single_text(text)
    print(f"\nPredicted Sentiment: {sentiment}")
    print(f"Confidence Score: {confidence:.2f}")
