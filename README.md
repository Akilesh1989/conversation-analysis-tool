# Conversation Analysis Tool

A comprehensive tool for analyzing conversations using sentiment analysis, text summarization, and question answering capabilities.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip
- virtualenv (recommended)

### Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/Akilesh1989/conversation-analysis-tool.git
cd conversation-analysis-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Components and Usage

### 1. Sentiment Analysis

The sentiment analysis component detects emotions in text using a transformer-based model.

#### Training the Model:
```bash
# Train the sentiment analysis model
python sentiment_analysis/training_with_xgboost.py
```

This will:
- Load the sample conversation dataset
- Generate embeddings using SentenceTransformer
- Train a classifier using XGBoost
- Save the trained model
  

You can also sentiment_analysis/training_with_multiple_models.py which will train the datasets with multiple models and save them in the models folder

#### Using Sentiment Analysis:
- Launch the Streamlit app
- Go to the "Sentiment Analysis" tab
- Enter text and click "Analyze Sentiment"
- View the detected sentiment and confidence score

### 2. Text Summarization

The summarization component creates concise summaries of longer texts.

#### Generating Summaries:

Be sure the change the NLTK_DATA_PATH in the summarization/main.py to your nltk_data path
```
NLTK_DATA_PATH = ''
```
in this file `summarization/main.py`

```bash
# Generate summaries for existing conversations
python summarization/main.py
```

This will:
- Process the sample conversations
- Generate summaries using extractive summarization
- Save summaries to text_summaries.txt

#### Using Summarization:
- Launch the Streamlit app
- Go to the "Summarization" tab
- View existing summaries
- You can also enter a new text, choose summary length (1-10 sentences)
- Click "Generate Summary"
- This will generate a summary

### 3. Question Answering (RAG)

The Q&A component uses Retrieval-Augmented Generation to answer questions about conversations.

#### Building the Index:
```bash
# Build the FAISS index for conversations
streamlit run q_and_a/build.py
```

This will:
- Process the conversation dataset
- Create text chunks
- Build a FAISS index for efficient retrieval
- Save the index and chunks

#### Using Q&A:
- Launch the Streamlit app
- Go to the "Q&A" tab
- Enter your question
- View the answer and referenced conversations

## 🖥️ Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501` with three main tabs:
1. Sentiment Analysis
2. Text Summarization
3. Question Answering

## 📁 Project Structure
```
conversation-analysis-tool/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Project dependencies
├── sentiment_analysis/           
│   ├── train4.py                 # Sentiment model training
│   ├── predict.py                # Sentiment prediction
│   └── report_generator.py       # Model performance reports
├── summarization/
│   ├── main.py                   # Summarization script
│   └── summarizer.py             # Summarization functions
└── q_and_a/
    ├── build.py                  # FAISS index building
    └── query.py                  # Question answering logic
```

## 🔍 Example Usage

### Command Line Interface:
```python
# Sentiment Analysis
from sentiment_analysis.predict import predict_single_text
sentiment, confidence = predict_single_text("I love this project!")

# Summarization
from summarization.summarizer import summarize_text
summary = summarize_text(long_text, num_sentences=3)

# Question Answering
from q_and_a.query import query_index
from q_and_a.build import load_faiss_data
index, chunks = load_faiss_data(500, 50)
response = query_index("What are the main topics?", index, chunks)
```

## 📝 Notes

- The sentiment analysis model supports 8 emotions: Happy, Sad, Angry, Surprised, Fearful, Disgusted, Curious, and Neutral
- Summarization works best on well-structured text with clear sentences
- Q&A performance depends on the quality and relevance of the indexed conversations

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
